import os

if __name__ == '__main__':
    ext_to_code_format = {
        '.py': 'Python',
        '.js': 'Js',
        '.c': 'C',
        '.cxx': 'Cpp',
        '.java': 'Java',
        '.sh': 'Bash',
        '.md': 'Markdown',
        '.yaml': 'Yaml',
        '.hs': 'Haskell',
        '.kt': 'Kotlin',
    }

    for root, dirs, files in os.walk('code_snippets'):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in ext_to_code_format:
                print(f'{file} | Expected: {ext_to_code_format[ext]}, Got: ')
                os.system(f'python main.py code_snippets/{file}')
                print()