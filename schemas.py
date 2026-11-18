"""Tool schema. The first sentence stays within 60 characters."""

SCHEMA = {
    "name": "cutout_png",
    "description": (
        "Cut a photo background and write a transparent PNG. "
        "Use this to make a photo or a product picture of a subject into one new transparent PNG "
        "for a game sprite or a prepared asset, with the backdrop knocked out and the foreground extracted. "
        "It does not resize or rotate the stored pixels."
    ),
    "annotations": {"readOnlyHint": False},
    "parameters": {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "Absolute path of the photo. Symlinks are not followed.",
            },
            "folder": {
                "type": "string",
                "description": "Absolute path of an existing folder. The plugin does not create it.",
            },
            "name": {
                "type": "string",
                "description": (
                    "Optional new PNG file name in that folder. "
                    "When omitted, the stem keeps ASCII letters, digits, dot, underscore, and hyphen. "
                    "An empty stem becomes photo-cutout.png. An existing name is refused."
                ),
            },
        },
        "required": ["image", "folder"],
    },
}
