import re, os


def read_file(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        context = f.read()
    return context

def get_files(files, path):
    if os.path.isdir(path):
        for folder in os.listdir(path):
            get_files(files, os.path.join(path, folder))
    elif path.endswith(".md"):
        files.append(path)
        
def fix_path(doc_root_path):
    files = []
    get_files(files, doc_root_path)
    file_map = {}
    for file in files:
        name = file.split("/")[-1]
        file_map[name] = "/" + file

    pattern = re.compile(r'\]\([^)]*\.md')
    for file in files:
        context = read_file(file)
        matches = pattern.findall(context)
        
        for match in matches:
            target = "](" + file_map[match.split("/")[-1].replace("](", "")]
            context = context.replace(match, target)
            print(match, target)
        
        with open(file, "w", encoding="utf-8") as f:
            f.write(context)

