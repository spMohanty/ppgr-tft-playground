from dataclasses import dataclass, fields

import click

def create_click_options(config_class):
    """Create click options automatically from a dataclass's fields."""
    def decorator(f):
        # Reverse the fields so the decorators are applied in the correct order
        for field in reversed(fields(config_class)):
            # Get the field type and default value
            param_type = field.type
            default_value = field.default
            
            # Convert type hints to click types
            type_mapping = {
                str: str,
                int: int,
                float: float,
                bool: bool
            }
            param_type = type_mapping.get(param_type, str)
            
            # Create the click option with both hyphen and underscore versions
            hyphen_name = f"--{field.name.replace('_', '-')}"
            underscore_name = f"--{field.name}"
            
            # Add aliases for specific parameters
            aliases = []
            if field.name == "debug_mode":
                aliases.append("--debug")
            
            # Combine all option names
            option_names = [hyphen_name, underscore_name] + aliases
            
            if param_type == bool:
                # Handle boolean flags differently
                f = click.option(
                    *option_names,
                    is_flag=True,
                    default=default_value,
                    help=f"Default: {default_value}"
                )(f)
            else:
                f = click.option(
                    *option_names,
                    type=param_type,
                    default=default_value,
                    help=f"Default: {default_value}"
                )(f)
        return f
    return decorator