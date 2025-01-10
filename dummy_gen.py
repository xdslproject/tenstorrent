import ast
import re


def get_class_info(file_path) -> list:
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read())

    class_infos = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_info = {
                "name": node.name
            }

            init_method = next(
                (n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None
            )

            class_info["args"] = []
            class_info["arg_types"] = []

            if init_method:
                for arg in init_method.args.args[1:]:  # Skip 'self'
                    class_info["args"].append(arg.arg)

                    # Check if the argument has a type hint
                    if isinstance(arg.annotation, ast.Name):
                        class_info["arg_types"].append(arg.annotation.id)

                    # Handle Union-like syntax for two values (e.g., A | B)
                    elif isinstance(arg.annotation, ast.BinOp) and isinstance(arg.annotation.op, ast.BitOr):
                        class_info["arg_types"].append(f"{arg.annotation.left.id} | {arg.annotation.right.id}")

                    else:
                        class_info["arg_types"].append(None)

            class_infos.append(class_info)

    return class_infos


def camel_to_snake(name):
    # Match sequences of capital letters and ensure separation from the rest
    snake_case = re.sub(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', '_', name)
    return snake_case.lower()


def mlir_to_python_type(type_hint):
    if type_hint == 'SSAValue | Operation':
        return 'int'

    if type_hint == 'IntegerAttr':
        return 'Literal[True, False]'

    else:
        raise ValueError(f'Unsupported type hint: {type_hint}')


if __name__ == "__main__":
    file_name = "tenstorrent/dialects/compute.py"

    classes = get_class_info(file_name)

    for class_def in classes:
        args = ""

        for arg_name, arg_type in zip(class_def["args"], class_def["arg_types"]):
            args += f"{arg_name}: {mlir_to_python_type(arg_type)}, "

        args = args[:-2]
        print(f"def {camel_to_snake(class_def['name'])}({args}):\n    pass")
        print()
        print()

