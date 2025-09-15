# arch_designer/core/simulator.py

import random
import os
import joblib
import pandas as pd
import subprocess
import json
import paramiko
from io import StringIO

class Simulator:
    def __init__(self, ml_model_path=None):
        """
        Ініціалізація: підвантажуємо ML-модель (якщо вказано),
        і визначаємо порядок колонок для DataFrame при ML-прогнозі.
        """
        self.model = None
        if ml_model_path and os.path.exists(ml_model_path):
            try:
                self.model = joblib.load(ml_model_path)
            except Exception as e:
                print(f"Warning: не вдалося завантажити ML-модель: {e}")

        self.feature_columns = [
            "pipeline_stages",
            "cache_inclusive",
            "cache_exclusive",
            "cache_size",
            "compute_units",
            "branch_dynamic",
            "out_of_order"
        ]

    def simulate(self, arch):
        """
        Псевдо-симуляція на основі простих формул.
        Повертає dict із ipc, energy, perf_per_watt[, perf_per_watt_ml].
        """
        # IPC
        ipc = arch.compute_units * 0.1 + arch.pipeline_stages * 0.05
        ipc *= 1 + arch.cache_size / 16
        ipc *= 1.2 if arch.out_of_order else 0.8
        ipc = round(ipc + random.uniform(-0.5, 0.5), 3)

        # Енергоспоживання
        energy = arch.compute_units * 2 + arch.pipeline_stages * 1.5
        energy *= 1.3 if arch.out_of_order else 1.0
        energy = round(energy + random.uniform(-5, 5), 3)

        # Продуктивність на ват
        perf_per_watt = round(ipc / energy, 5)

        metrics = {
            "ipc": ipc,
            "energy": energy,
            "perf_per_watt": perf_per_watt
        }

        # ML-прогноз (якщо завантажена модель)
        if self.model:
            feat = {
                "pipeline_stages": arch.pipeline_stages,
                "cache_inclusive": 1 if arch.cache_type == "Inclusive" else 0,
                "cache_exclusive": 1 if arch.cache_type == "Exclusive" else 0,
                "cache_size": arch.cache_size,
                "compute_units": arch.compute_units,
                "branch_dynamic": 1 if arch.branch_predictor == "Dynamic" else 0,
                "out_of_order": 1 if arch.out_of_order else 0
            }
            df_feat = pd.DataFrame([feat], columns=self.feature_columns)
            try:
                ml_pred = self.model.predict(df_feat)[0]
                metrics["perf_per_watt_ml"] = round(float(ml_pred), 5)
            except Exception as e:
                print(f"Warning: ML-прогноз не вдався: {e}")

        return metrics

    def simulate_remote_ssh(self,
                            arch,
                            host, port, username, password,
                            remote_base="~/arch_sandbox",
                            gem5_repo="https://gem5.googlesource.com/public/gem5",
                            gem5_dir="gem5",
                            config_script_local="configs/my_cpu_config.py"):
        """
        Віддалена симуляція через SSH:
        1) Підключення до host:port
        2) Створення тек {remote_base}, {remote_base}/configs, {remote_base}/out
        3) Встановлення Python-залежностей та збірка gem5 (якщо немає)
        4) Завантаження локального скрипта configs/my_cpu_config.py → remote_base/configs/
        5) Завантаження архітектурного JSON → remote_base/arch.json
        6) Виконання gem5.opt із вказаними параметрами
        7) Читання {remote_base}/out/stats.txt і повернення метрик + журналу
        """
        logs = []

        # 1) Налагоджуємо SSH-клієнт
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, port=port, username=username, password=password)
        sftp = ssh.open_sftp()

        # Розгортаємо remote_base
        def ensure_dir(path):
            cmd = f"mkdir -p {path}"
            ssh.exec_command(cmd)

        remote_base = remote_base.rstrip('/')
        base = remote_base
        conf_dir = f"{base}/configs"
        out_dir  = f"{base}/out"
        gem5_path = f"{base}/{gem5_dir}/build/X86/gem5.opt"

        logs.append(f"Створюємо теку {base}")
        ensure_dir(base)
        logs.append(f"Створюємо теку {conf_dir}")
        ensure_dir(conf_dir)
        logs.append(f"Створюємо теку {out_dir}")
        ensure_dir(out_dir)

        # 2) Встановлюємо Python-залежності
        deps = "streamlit pandas scikit-learn joblib deap matplotlib paramiko"
        logs.append("Встановлюємо Python-залежності (pip)")
        stdin, stdout, stderr = ssh.exec_command(f"pip3 install --user {deps}")
        logs.extend([line for line in stdout.read().decode().splitlines()])
        logs.extend([line for line in stderr.read().decode().splitlines()])

        # 3) Клонуємо / збираємо gem5, якщо ще немає
        gem5_remote = f"{base}/{gem5_dir}"
        logs.append(f"Перевірка наявності gem5 у {gem5_remote}")
        stdin, stdout, stderr = ssh.exec_command(f"if [ ! -d {gem5_remote} ]; then git clone {gem5_repo} {gem5_remote}; fi")
        logs.append(stdout.read().decode())
        logs.append(stderr.read().decode())

        logs.append("Збираємо gem5 (scons build/X86/gem5.opt)")
        cmd_build = f"cd {gem5_remote} && scons build/X86/gem5.opt -j$(nproc)"
        stdin, stdout, stderr = ssh.exec_command(cmd_build)
        logs.extend([line for line in stdout.read().decode().splitlines()])
        logs.extend([line for line in stderr.read().decode().splitlines()])

        # 4) Завантажуємо локальний config_script
        remote_config = f"{conf_dir}/my_cpu_config.py"
        logs.append(f"Завантажуємо скрипт-конфіг → {remote_config}")
        sftp.put(config_script_local, remote_config)

        # 5) Завантажуємо архітектуру у JSON
        arch_json = json.dumps(arch.to_dict())
        with sftp.open(f"{base}/arch.json", "w") as f:
            f.write(arch_json)
        logs.append("Завантажили arch.json")

        # 6) Очищаємо out_dir і запускаємо симуляцію
        ssh.exec_command(f"rm -rf {out_dir}/*")
        logs.append("Очистили out_dir")

        cmd_sim = f"{gem5_path} {remote_config} --arch-json {base}/arch.json --outdir {out_dir}"
        logs.append(f"Запуск симуляції: {cmd_sim}")
        stdin, stdout, stderr = ssh.exec_command(cmd_sim)
        # чекаємо на завершення
        exit_status = stdout.channel.recv_exit_status()
        logs.append(f"Статус виконання gem5: {exit_status}")
        logs.extend([line for line in stdout.read().decode().splitlines()])
        logs.extend([line for line in stderr.read().decode().splitlines()])

        # 7) Парсимо stats.txt
        stats_file = f"{out_dir}/stats.txt"
        logs.append(f"Читаємо {stats_file}")
        with sftp.open(stats_file) as f:
            stats_lines = f.read().splitlines()

        ssh.close()
        sftp.close()

        ipc = energy = None
        for line in stats_lines:
            if line.startswith("system.cpu.commit.Instructions"):
                ipc = float(line.split()[1])
            if line.startswith("system.energy.total_energy"):
                energy = float(line.split()[1])

        if ipc is None or energy is None:
            raise RuntimeError("Не знайдено ipc або energy у stats.txt")

        perf_per_watt = round(ipc / energy, 5)
        metrics = {
            "ipc": ipc,
            "energy": energy,
            "perf_per_watt": perf_per_watt
        }

        return metrics, logs
