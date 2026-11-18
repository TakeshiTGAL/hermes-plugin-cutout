"""Cut a photo background into one new transparent PNG."""

from pathlib import Path

_SKILL_DESCRIPTION = (
    "Cut a photo background into one new transparent PNG in a folder you name."
)


def register(ctx) -> None:
    if __package__:
        from . import delivery, gates
        from .cli import handle, setup_parser
        from .schemas import SCHEMA
        from .service import cutout_png, on_pre_tool_call
    else:
        import delivery
        import gates
        from cli import handle, setup_parser
        from schemas import SCHEMA
        from service import cutout_png, on_pre_tool_call

    gates.remember_context(ctx)
    ctx.register_tool(
        name="cutout_png",
        toolset="cutout",
        schema=SCHEMA,
        handler=cutout_png,
    )
    ctx.register_skill(
        "cutout",
        Path(__file__).resolve().parent / "skills" / "cutout" / "SKILL.md",
        description=_SKILL_DESCRIPTION,
    )
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    register_hook = getattr(ctx, "register_hook", None)
    if callable(register_hook):
        try:
            register_hook("pre_llm_call", delivery.start_turn)
            register_hook("transform_llm_output", delivery.attach_reply)
        except Exception:
            pass
    ctx.register_cli_command(
        name="cutout",
        help="Print a background-cut plan as JSON",
        setup_fn=setup_parser,
        handler_fn=handle,
        description="Print a cutout plan. This command does not write a PNG.",
    )
