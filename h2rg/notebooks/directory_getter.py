import os

def main():
    root_dir = '/proj/case/2025-06-05'
    all_dirs = os.listdir(root_dir)

    directories = []


    for curr_dir in all_dirs:
        if 'Euclid' in curr_dir:
            directories.append(f'{root_dir}/{curr_dir}')
        elif 'FPM' in curr_dir:
            nested_dir = os.listdir(f'{root_dir}/{curr_dir}')[0]

            directories.append(f'{root_dir}/{curr_dir}/{nested_dir}')

    for directory in directories:
        print(directory)

    print(len(directories))


if __name__ == '__main__':
    main()