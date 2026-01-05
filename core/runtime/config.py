# ==========================================
# CONFIGURATION & UTILITIES
# ==========================================

def resolve_refs(schema):
    """Resolve $ref references in JSON schema to inline definitions."""
    defs = schema.get('$defs', {})

    def expand(node):
        if isinstance(node, dict):
            # If it's a reference, replace it with the actual definition
            if '$ref' in node:
                ref_name = node['$ref'].split('/')[-1]
                # Recursively expand the definition we found
                return expand(defs[ref_name])
            
            # Otherwise, traverse dict, removing '$defs' keys
            return {
                k: expand(v) 
                for k, v in node.items() 
                if k != '$defs'
            }
        elif isinstance(node, list):
            return [expand(item) for item in node]
        else:
            return node

    return expand(schema)


def load_model_config(model_name):
    """Load model configuration from models.json registry."""
    try:
        paths = ["models.json", os.path.expanduser("~/.enso/models.json")]
        registry = {}
        for p in paths:
            if os.path.exists(p):
                with open(p, "r") as f:
                    registry = json.load(f)
                break
        if model_name in registry:
            return registry[model_name]
        return {"type": "openai", "cost_in": 0, "cost_out": 0}
    except Exception:
        return {"type": "openai", "cost_in": 0, "cost_out": 0}
