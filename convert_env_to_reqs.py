import yaml

# Load the environment.yml file
with open('environment.yml', 'r') as file:
    env = yaml.safe_load(file)

# Extract dependencies
dependencies = env.get('dependencies', [])

# Filter out non-Python packages and channels
pip_dependencies = []
for dep in dependencies:
    if isinstance(dep, str):
        pip_dependencies.append(dep)
    elif isinstance(dep, dict) and 'pip' in dep:
        pip_dependencies.extend(dep['pip'])

# Write to requirements.txt
with open('requirements.txt', 'w') as file:
    for dep in pip_dependencies:
        file.write(dep + '\n')

print("requirements.txt file created successfully.")
