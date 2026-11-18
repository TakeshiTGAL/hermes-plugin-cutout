---
name: cutout
description: Cut a photo background into one new transparent PNG in a folder you name.
---

Use tool_search before web_search to cut a photo background into a transparent PNG.

Before approval, write in the message the photo path, the folder, the new PNG name, and that the file is new and an existing file is not overwritten.

## Ordinary job

When someone wants the background of a photo removed for a game asset or a product photo, call cutout_png with the absolute photo path and an existing folder. Pass name only when they asked for a file name. The tool asks for approval, then writes one new PNG. It does not resize or rotate the stored pixels. It does not overwrite an existing file. When the result contains a media line, copy that whole line onto its own line in the reply. That is how the chat shows the PNG. A Telegram media line starts with `[[as_document]]`. Keep that marker. A terminal result has no media line, and an inherited `HERMES_SESSION_SOURCE` does not add one. Give the path, and do not write a `MEDIA:` line.

After the plugin is enabled, the command `hermes cutout call` prints JSON and does not write. Exit 2 means nothing was written. Approving cutout_png in chat is what writes the file. Approval is the `pre_tool_call` hook, and that hook returns `approve`. When install omits `--yes-deps` in a shell that is not a terminal, cutout stays disabled and `hermes cutout` is not a hermes command. That invocation prints no JSON. Turn the tool off on one surface with `hermes tools disable cutout --platform cli`. There is no platform named gateway.

## What this plugin does not do

It does not upload the photo. It does not call a remote background service. The first run downloads the pinned u2netp file when HERMES_HOME/models/cutout/u2netp.onnx is absent, checks sha256 309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8, and stores those 4574861 bytes. Later runs read that file and check the hash again. A mismatch is not loaded and is not replaced.

It does not use the rembg default model bria-rmbg, and it does not download that model. It does not resize or rotate. Dimension changes belong to image-utils.

Yolo and approvals off skip the card. Always applies to that same plan only, and only while a person is present. Hermes adds one plugin_rule line to command_allowlist in config.yaml. The line stores the plan digest and does not store the photo path. The next call with the same arguments skips the card and writes the PNG. A different folder is a different line and asks again. Cron and other unattended runs stay refused even after always. plugins.isolation host refuses the write. Anyone on your gateway can call the tool. The model file stays after uninstall. The dependency environment under HERMES_HOME/installs stays too.

## When it fails

- not_image: Pass a photo file. Nothing was written.
- too_large: The photo is over 33554432 bytes or 24000000 pixels. Pass a smaller photo.
- model_fetch: The pinned download failed. Check the network and call again. The photo was not uploaded.
- sha256_mismatch: When HERMES_HOME/models/cutout/u2netp.onnx is already on disk and the hash does not match, delete that file and call again. When the download does not match, nothing is saved and there is no file to delete. Call again. The bytes were not loaded.
- exists: Choose a new name. An existing file is not overwritten.
- symlink: The photo, the folder, the destination, or a parent directory is a symlink. Pass the real path. A symlink was not followed. The reply names that directory, including letters outside ASCII. A control character, a format character, a backtick, a slash, a blank name, or a name longer than 80 characters is called that directory.
- outside: Pass an absolute folder and a single PNG file name.
- hardlink: Choose a name that is not a hard link.
- missing: Create the folder, or pass a photo that exists. The plugin does not create the folder.
- partial: The new PNG was not confirmed. Delete that name if you do not want it, and call again.
- unreadable: Check the file permissions and call again.
- dependency: Install the plugin dependencies and call again.
- model_runtime: The local model did not run. Another model is not selected.
- approval_denied: Call cutout_png again and approve that card.
- approval_required: Approve cutout_png in chat. The command does not write.
- approval_not_sent: Use a shorter folder path and call again.
- approval_blocked: Run the plugin outside plugins.isolation host.
- unattended: Run cutout_png from a chat you can answer.
- bad_argument: Pass image, folder, and an optional png name.
