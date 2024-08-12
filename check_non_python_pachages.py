import yaml

# Load the environment.yml file
with open('environment.yml', 'r') as file:
    env = yaml.safe_load(file)

# Extract dependencies
dependencies = env.get('dependencies', [])

# List to hold non-Python packages
non_python_packages = []

# Check each dependency
for dep in dependencies:
    if isinstance(dep, str):
        # Check for typical non-Python package identifiers
        if any(prefix in dep for prefix in ['lib', 'gcc', 'openmpi', 'ffmpeg']):
            non_python_packages.append(dep)
    elif isinstance(dep, dict) and 'pip' not in dep:
        non_python_packages.extend(dep.keys())

# Print the non-Python packages
if non_python_packages:
    print("Non-Python packages found in environment.yml:")
    for pkg in non_python_packages:
        print(f"- {pkg}")
else:
    print("No non-Python packages found in environment.yml.")
