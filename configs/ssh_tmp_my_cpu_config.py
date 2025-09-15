# configs/my_cpu_config.py

import json
import argparse
from m5.objects import *
import m5

def main():
    parser = argparse.ArgumentParser(description="gem5 CPU config from JSON")
    parser.add_argument("--arch-json", required=True, help="Шлях до JSON з параметрами архітектури")
    parser.add_argument("--outdir",    required=True, help="Тека для результатів (stats.txt)")
    args = parser.parse_args()

    # 1) Зчитуємо архітектуру
    with open(args.arch_json) as f:
        arch = json.load(f)

    # 2) Будуємо систему
    system = System()
    # Встановлюємо клочну домену
    system.clk_domain = SrcClockDomain(
        clock='2GHz',
        voltage_domain=VoltageDomain()
    )
    system.mem_mode = 'timing'
    system.mem_ranges = [AddrRange('512MB')]

    # 3) Налаштовуємо CPU
    cpu = DerivO3CPU()
    # Використовуємо pipeline_stages як fetch-to-decode delay
    cpu.fetchToDecodeDelay = arch.get("pipeline_stages", 4)
    # За бажанням тут можна розгорнути arch["cache_size"] і arch["cache_type"]
    system.cpu = cpu

    # 4) Створюємо корінь і ініціалізуємо
    root = Root(full_system=False, system=system)
    m5.instantiate()

    # 5) Запускаємо симуляцію
    print("=== Running gem5 simulation ===")
    exit_event = m5.simulate()
    print(f"=== Exiting @ tick {m5.curTick()} because {exit_event.getCause()} ===")

    # 6) Записуємо stats.txt
    stats_file = f"{args.outdir}/stats.txt"
    with open(stats_file, "w") as out:
        for stat in m5.stats.elements():
            out.write(f"{stat.name} {stat.value}\n")
    print(f"=== Stats saved to {stats_file} ===")

if __name__ == "__main__":
    main()
