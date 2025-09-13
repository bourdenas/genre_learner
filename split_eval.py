import csv
import random
import sys

def split_csv(input_file, output_file_70, output_file_30):
    with open(input_file, newline='') as csvfile:
        reader = list(csv.reader(csvfile))
        header = reader[0]
        rows = reader[1:]

    random.shuffle(rows)
    split_index = int(0.7 * len(rows))
    rows_70 = rows[:split_index]
    rows_30 = rows[split_index:]

    with open(output_file_70, 'w', newline='') as f70:
        writer = csv.writer(f70)
        writer.writerow(header)
        writer.writerows(rows_70)

    with open(output_file_30, 'w', newline='') as f30:
        writer = csv.writer(f30)
        writer.writerow(header)
        writer.writerows(rows_30)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python split_eval.py <input.csv> <output_70.csv> <output_30.csv>")
        sys.exit(1)
    split_csv(sys.argv[1], sys.argv[2], sys.argv[3])
