import argparse
import json
import csv
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
import task_01

# Запускаем 1 задачу для создания файлов
task_01.main("./")


def read_txt_file(filename):
    """Читает данные из TXT файла (формат: x    y)"""
    x_values = []
    y_values = []

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        x_values.append(x)
                        y_values.append(y)
                    except ValueError:
                        print(f"Предупреждение: Не удалось распознать строку {line_num}: {line}")
                        continue
                else:
                    print(f"Предупреждение: Неверный формат строки {line_num}: {line}")

    except FileNotFoundError:
        print(f"Ошибка: Файл {filename} не найден")
        return None, None
    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")
        return None, None

    return np.array(x_values), np.array(y_values)


def read_csv_file(filename):
    """Читает данные из CSV файла (формат: номер, x, y)"""
    x_values = []
    y_values = []

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row_num, row in enumerate(reader, 1):
                if len(row) >= 3:
                    try:
                        # Пропускаем первый столбец (номер строки)
                        x = float(row[1])
                        y = float(row[2])
                        x_values.append(x)
                        y_values.append(y)
                    except (ValueError, IndexError):
                        print(f"Предупреждение: Не удалось распознать строку {row_num}: {row}")
                        continue
                else:
                    print(f"Предупреждение: Неверный формат строки {row_num}: {row}")

    except FileNotFoundError:
        print(f"Ошибка: Файл {filename} не найден")
        return None, None
    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")
        return None, None

    return np.array(x_values), np.array(y_values)


def read_json_file(filename):
    """Читает данные из JSON файла (поддерживает оба формата)"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Формат 1: {"x": [...], "y": [...]}
        if "x" in data and "y" in data:
            x_values = np.array(data["x"])
            y_values = np.array(data["y"])
            return x_values, y_values

        # Формат 2: {"data": [{"x": x1, "y": y1}, ...]}
        elif "data" in data:
            x_values = []
            y_values = []
            for point in data["data"]:
                if "x" in point and "y" in point:
                    x_values.append(float(point["x"]))
                    y_values.append(float(point["y"]))
            return np.array(x_values), np.array(y_values)

        else:
            print("Ошибка: Неподдерживаемый формат JSON файла")
            return None, None

    except FileNotFoundError:
        print(f"Ошибка: Файл {filename} не найден")
        return None, None
    except json.JSONDecodeError as e:
        print(f"Ошибка при разборе JSON файла: {e}")
        return None, None
    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")
        return None, None


def read_xml_file(filename):
    """Читает данные из XML файла (поддерживает оба формата)"""
    try:
        tree = ET.parse(filename)
        root = tree.getroot()

        # Формат 1: отдельные секции xdata и ydata
        xdata = root.find('xdata')
        ydata = root.find('ydata')

        if xdata is not None and ydata is not None:
            x_values = []
            y_values = []

            # Читаем x значения (пропускаем первый элемент - количество точек)
            x_elements = xdata.findall('x')
            for x_elem in x_elements:
                try:
                    x_values.append(float(x_elem.text))
                except ValueError:
                    continue

            # Читаем y значения
            y_elements = ydata.findall('y')
            for y_elem in y_elements:
                try:
                    y_values.append(float(y_elem.text))
                except ValueError:
                    continue

            return np.array(x_values), np.array(y_values)

        # Формат 2: строки с парами x, y
        else:
            x_values = []
            y_values = []

            rows = root.findall('row')
            for row in rows:
                x_elem = row.find('x')
                y_elem = row.find('y')

                if x_elem is not None and y_elem is not None:
                    try:
                        x = float(x_elem.text)
                        y = float(y_elem.text)
                        x_values.append(x)
                        y_values.append(y)
                    except ValueError:
                        continue

            return np.array(x_values), np.array(y_values)

    except FileNotFoundError:
        print(f"Ошибка: Файл {filename} не найден")
        return None, None
    except ET.ParseError as e:
        print(f"Ошибка при разборе XML файла: {e}")
        return None, None
    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")
        return None, None


def read_data_file(filename):
    """Автоматически определяет формат файла и читает данные"""
    file_ext = Path(filename).suffix.lower()

    if file_ext == '.txt':
        return read_txt_file(filename)
    elif file_ext == '.csv':
        return read_csv_file(filename)
    elif file_ext == '.json':
        return read_json_file(filename)
    elif file_ext == '.xml':
        return read_xml_file(filename)
    else:
        print(f"Ошибка: Неподдерживаемый формат файла: {file_ext}")
        return None, None


def create_plot(x_values, y_values, args):
    """Создает и отображает график"""
    # Создаем фигуру
    plt.figure(figsize=(12, 8))

    # Строим график
    plt.plot(x_values, y_values, 'b-', linewidth=1.5, label='f(x)')

    # Настраиваем оси
    if args.xmin is not None and args.xmax is not None:
        plt.xlim(args.xmin, args.xmax)
    elif args.xmin is not None:
        plt.xlim(left=args.xmin)
    elif args.xmax is not None:
        plt.xlim(right=args.xmax)

    if args.ymin is not None and args.ymax is not None:
        plt.ylim(args.ymin, args.ymax)
    elif args.ymin is not None:
        plt.ylim(bottom=args.ymin)
    elif args.ymax is not None:
        plt.ylim(top=args.ymax)

    # Подписи осей
    plt.xlabel(args.xlabel, fontsize=12)
    plt.ylabel(args.ylabel, fontsize=12)

    # Заголовок
    plt.title(args.title, fontsize=14, pad=20)

    # Сетка
    if args.grid:
        plt.grid(True, which='major', alpha=0.5)
        plt.grid(True, which='minor', alpha=0.3)
        plt.minorticks_on()

    # Легенда
    if args.legend:
        plt.legend(fontsize=12)

    # Сохранение или показ
    if args.output:
        plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
        print(f"График сохранен в файл: {args.output}")

    if args.show:
        plt.show()


def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='Отображение графиков y = f(x) на основе файлов данных',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Примеры использования:
  python task_02.py input.json
  python task_02.py input.json --xmin=-10 --xmax=10
  python task_02.py input.xml --title "График функции"
  python task_02.py input.txt --output "result.png"
  python task_02.py input.csv --xlabel "Аргумент x" --ylabel "Значение f(x)"
        '''
    )

    # Обязательный аргумент - файл с данными
    parser.add_argument('filename', help='Имя файла с данными (TXT, CSV, JSON, XML)')

    # Параметры осей
    parser.add_argument('--xmin', type=float, help='Минимальное значение по оси X')
    parser.add_argument('--xmax', type=float, help='Максимальное значение по оси X')
    parser.add_argument('--ymin', type=float, help='Минимальное значение по оси Y')
    parser.add_argument('--ymax', type=float, help='Максимальное значение по оси Y')

    # Подписи
    parser.add_argument('--xlabel', default='x', help='Подпись по оси X (по умолчанию: "x")')
    parser.add_argument('--ylabel', default='f(x)', help='Подпись по оси Y (по умолчанию: "f(x)")')
    parser.add_argument('--title', default='График функции y = f(x)',
                        help='Заголовок графика')

    # Настройки отображения
    parser.add_argument('--grid', action='store_true', default=True,
                        help='Отображать сетку (по умолчанию: включена)')
    parser.add_argument('--no-grid', dest='grid', action='store_false',
                        help='Не отображать сетку')
    parser.add_argument('--legend', action='store_true', default=True,
                        help='Отображать легенду (по умолчанию: включена)')
    parser.add_argument('--no-legend', dest='legend', action='store_false',
                        help='Не отображать легенду')

    # Сохранение
    parser.add_argument('--output', '-o', help='Имя файла для сохранения графика')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Разрешение для сохранения (по умолчанию: 300)')

    # Отображение
    parser.add_argument('--show', action='store_true', default=True,
                        help='Показать график на экране (по умолчанию: включено)')
    parser.add_argument('--no-show', dest='show', action='store_false',
                        help='Не показывать график на экране')

    return parser.parse_args()


def print_data_info(x_values, y_values, filename):
    """Выводит информацию о загруженных данных"""
    print(f"Загружен файл: {filename}")
    print(f"Количество точек: {len(x_values)}")
    print(f"Диапазон x: [{np.min(x_values):.6f}, {np.max(x_values):.6f}]")
    print(f"Диапазон y: [{np.min(y_values):.10f}, {np.max(y_values):.10f}]")
    print()


def main():
    """Основная функция программы"""
    args = parse_arguments()

    # Проверяем существование файла
    if not os.path.exists(args.filename):
        print(f"Ошибка: Файл {args.filename} не найден")
        sys.exit(1)

    # Читаем данные
    print("Загрузка данных...")
    x_values, y_values = read_data_file(args.filename)

    if x_values is None or y_values is None:
        print("Ошибка при загрузке данных")
        sys.exit(1)

    if len(x_values) == 0 or len(y_values) == 0:
        print("Ошибка: Файл не содержит данных")
        sys.exit(1)

    if len(x_values) != len(y_values):
        print(f"Предупреждение: Количество x и y значений не совпадает ({len(x_values)} vs {len(y_values)})")
        # Берем минимальное количество
        min_len = min(len(x_values), len(y_values))
        x_values = x_values[:min_len]
        y_values = y_values[:min_len]

    # Выводим информацию о данных
    print_data_info(x_values, y_values, args.filename)

    # Создаем график
    print("Создание графика...")
    create_plot(x_values, y_values, args)



if __name__ == "__main__":
    main()