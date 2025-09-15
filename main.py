import json
import click
from core.architecture import Architecture
from core.simulator import Simulator
from core.optimizer import Optimizer
from core.visualizer import plot_ga_log

@click.group()
def cli():
    """arch_designer — генератор и оценщик микроархитектур."""
    pass

@cli.command()
@click.option('--arch-type',    required=True, help='Тип архитектуры: CPU/GPU/DSP/FPGA/ASIC')
@click.option('--pipeline',     required=True, type=int, help='Количество стадий конвейера')
@click.option('--cache-type',   required=True, help='Тип кеша: Inclusive/Exclusive/Non-inclusive')
@click.option('--cache-size',   required=True, type=int, help='Размер кеша в МБ')
@click.option('--compute-units',required=True, type=int, help='Число вычислительных единиц')
@click.option('--branch-predictor', required=True, help='Static или Dynamic')
@click.option('--out-of-order', is_flag=True, help='Флаг Out-of-Order исполнения')
def manual(arch_type, pipeline, cache_type, cache_size, compute_units, branch_predictor, out_of_order):
    """Ручной ввод параметров архитектуры."""
    sim = Simulator()
    arch = Architecture(arch_type, pipeline, cache_type, cache_size,
                        compute_units, branch_predictor, out_of_order)
    m = sim.simulate(arch)
    click.echo(arch)
    click.echo(json.dumps(m, indent=2))

@cli.command()
@click.option('--count', default=5, help='Сколько случайных архитектур сгенерировать')
def random(count):
    """Автоматическая рандомная генерация."""
    sim = Simulator()
    out = []
    for _ in range(count):
        arch = Architecture.random_architecture()
        m = sim.simulate(arch)
        out.append({'arch': arch.to_dict(), 'metrics': m})
    click.echo(json.dumps(out, indent=2))

@cli.command()
@click.option('--model-path', default='models/efficiency_model.pkl', help='Путь к ML-модели')
def ml(model_path):
    """Оценка через ML-модель."""
    sim = Simulator(ml_model_path=model_path)
    arch = Architecture.random_architecture()
    m = sim.simulate(arch)
    click.echo(arch)
    click.echo(json.dumps(m, indent=2))

@cli.command()
@click.option('--pop-size', default=50, help='Размер популяции для GA')
@click.option('--ngen',     default=40, help='Число поколений для GA')
@click.option('--cxpb',     default=0.5, help='Вероятность скрещивания')
@click.option('--mutpb',    default=0.2, help='Вероятность мутации')
def ga(pop_size, ngen, cxpb, mutpb):
    """Генетический алгоритм поиска оптимальной архитектуры."""
    sim = Simulator()
    opt = Optimizer(sim)
    best_arch, best_metrics, log = opt.genetic_algorithm(
        pop_size=pop_size, ngen=ngen, cxpb=cxpb, mutpb=mutpb
    )
    click.echo("=== BEST ARCHITECTURE ===")
    click.echo(best_arch)
    click.echo("=== METRICS ===")
    click.echo(json.dumps(best_metrics, indent=2))
    plot_ga_log(log)

if __name__ == '__main__':
    cli()
