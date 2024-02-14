import json

with open("../../concepts.json", "r") as f:
    data = json.load(f)

for storyboard in data:
    for frame, details in storyboard["implementation"].items():
        description = details["description"]
        with open(f"storyboard_{storyboard['concept']}_{frame}.txt", "w") as f:
            f.write(description)
