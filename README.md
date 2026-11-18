# Cutout

![Before and after. Left is the photo. Right is the PNG this plugin wrote, shown on gray.](assets/before-after.png)

Photograph by Rachel Michetti, CC0. scikit-image v0.25.2, in `skimage/data/_fetchers.py`, says: "No copyright restrictions.  CC0 by the photographer (Rachel Michetti)." File: https://github.com/scikit-image/scikit-image/raw/v0.25.2/skimage/data/coffee.png . Note: https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/_fetchers.py

The left half is that photo at its stored size, 600 by 400. No person is in the frame. The right half is the PNG the pinned weights wrote from that photo, placed on gray (230, 230, 230) so the transparent pixels show. Those cutout pixels were not repainted. The weights were u2netp v0.0.0, sha256 `309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8`.

A photo and a folder → one new transparent PNG in that folder. A chat that can attach a file receives that PNG as an image. A terminal receives the path.

Ask in chat to cut the background of `/photos/box.png` into `/photos/out` → `cutout_png` asks for approval, then writes one new PNG. The stored pixel size stays as it is. On Desktop, Discord, Slack, and the other chats that attach a file, the result includes a `media` line, `MEDIA:` plus the absolute path of that PNG. The reply carries the same line, and the chat shows the file. When Hermes calls `transform_llm_output`, the plugin adds a line for each PNG this turn wrote that the reply left out, other than a PNG written between two session changes in that turn, and it does not repeat a line the reply already has. A line counts as already written only when Hermes would attach that path. A fenced code block, a blockquote, or a JSON string does not count. Inline code counts when the whole text inside the backticks is a `MEDIA:` line and that path exists, because Hermes attaches it. It does not add a line for a PNG an earlier turn wrote. A `/btw` answer and a background review share the session and do not receive the line. The turn reply does. With `compression.in_place: false`, one session change in a turn still puts the line on that reply. When the session changes twice in one turn, a PNG written between those two changes gets no line on the reply. That PNG stays in the folder you named. The default `in_place: true` keeps the line on that reply. When Hermes ends the turn with an empty final reply, it does not call `transform_llm_output`, so no line is added. The PNG stays in the folder you named. The next turn's reply does not gain the line. That empty final reply was checked on Hermes 0.22.2 and 0.23.1 when an empty body continued until the repeat limit. On Telegram the line starts with `[[as_document]]`, and Hermes sends the original PNG as a file. A photo send of a large image composites transparent pixels onto white and stores a JPEG. Whether Discord or another chat keeps those pixels on screen was not checked, so this page does not say that they do. A terminal (CLI or TUI), SMS, ACP, and the local gateway have no attachment channel. Those results stay the path and the width and the height. They do not include a `MEDIA:` line, and the reply does not gain one. An inherited `HERMES_SESSION_SOURCE` does not turn a terminal into Desktop. Desktop receives the line because that session binds the source. `api_server`, `webhook`, and `msgraph_webhook` are refused before a write, so nothing is written and no path is returned. The Hermes `api_server` chat, completions, and responses endpoints can inline a `MEDIA:` image as a data URL. This plugin does not write on `api_server`, so that path is not used.

Create `photo.png` and a folder named `out` in this directory, then run:

```
hermes plugins install "file://$PWD" --enable --yes-deps
hermes cutout call --image "$PWD/photo.png" --folder "$PWD/out"
```

`hermes plugins enable cutout` is the step when install did not use `--enable`. A shell that is not a terminal skips Python dependency consent when `--yes-deps` is absent. The plugin then stays disabled, and `hermes cutout` is not a hermes command. That invocation prints no JSON. After cutout is enabled, the call command prints one JSON object and exits 2. It does not write the PNG. Approving `cutout_png` in chat writes one new file. Approval is `pre_tool_call` returning `approve`.

## Details

Linux and macOS are the listed platforms. The tool reads the photo on this machine and writes one new PNG in the folder you named. The plugin does not create that folder. An existing file is not overwritten, including a different case on a case-insensitive disk when the name is already there. A symlink is not followed. That check covers the photo file, the folder, the destination name, and every parent directory. A parent directory that is a symlink is refused, and the reply names that directory, including letters outside ASCII. A control character, a format character, a backtick, a slash, a blank name, or a name longer than 80 characters is called that directory. A regular file reached only through that directory is refused too. On macOS, `/tmp` is a symlink. A regular photo whose path goes through that symlink is refused until you pass the real directory. A hard link at the destination is refused. `..` is refused.

The photo is not uploaded. The plugin does not call a remote background service. Resize and rotate stay with image-utils. EXIF orientation is not applied and is not copied. An animated file contributes its first frame only, and the reply includes `frame_count`. The stored width and height are the output width and height. The weights use a 320 pixel working mask and scale that mask back to the stored size.

The pinned weights are u2netp `v0.0.0`, 4574861 bytes, sha256 `309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8`, from `https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx`. The file is stored at `HERMES_HOME/models/cutout/u2netp.onnx`. The first run downloads it when that file is absent. Later runs hash the file again and do not download. A file already at that path whose hash does not match is not loaded and is not replaced. Delete that file yourself before a fresh download. A download whose hash does not match is not saved, and the reply does not ask you to delete a file.

The weights are the small u2netp file published with rembg `v0.0.0`. The upstream project is `https://github.com/xuebinqin/U-2-Net` and its license file is `https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE` (Apache-2.0). rembg itself is MIT at `https://github.com/danielgatis/rembg`. The rembg maintainer wrote that the MIT license does not cover the weights: `https://github.com/danielgatis/rembg/issues/837`. The rembg default model name is bria-rmbg, and this plugin does not download it. Edges from u2netp are coarse, and hair or a thin object can be missed. `alpha_zero` and `alpha_full` count pixels in the file that was written. They are not a promise that the background is gone.

A photo over 33554432 bytes, or over 24000000 pixels, is refused before a write. A file that is not an image is refused. The download cap for the weights is 8388608 bytes. The client allows one HTTPS redirect from `github.com` to `release-assets.githubusercontent.com`, with no proxy environment variables and no Authorization header.

Hermes installs `rembg>=2.0.87,<3` and `onnxruntime>=1.31.1,<2`. Measured on 2026-11-10, the rembg 2.0.87 wheel is 70625 bytes. The onnxruntime 1.31.1 CPU wheels for Linux and macOS run from 20871676 to 23773876 bytes. Windows CPU wheels for 1.31.1 are smaller and sit below that range. The macOS file is the arm64 wheel tagged `macosx_14_0`. PyPI lists no Intel macOS wheel for 1.31.1, so dependency install does not succeed on that machine. rembg also installs jsonschema, numpy, pillow, pooch, pymatting, scikit-image, scipy, and tqdm. One selection of those wheels (macOS arm64 where a platform wheel was current, otherwise the pure-Python wheel) is jsonschema 4.26.0 at 90630 bytes, numpy 2.5.4 at 12015718, pillow 13.0.0 at 4852329, pooch 1.9.0 at 67175, pymatting 1.1.16 at 56257, scikit-image 0.26.0 at 12061567, scipy 1.18.1 at 28733332, and tqdm 4.70.1 at 80199. That selection sums to 57957207 bytes. The versions in that selection are one example from 2026-11-10. The installer can resolve other versions inside the declared ranges. Other platforms differ. The GPU package is not a dependency.

`cutout_png` is the only tool, and it is the only one that asks for approval. The card names the photo path, the new PNG path, the sha256, and that the photo is not uploaded. The same arguments are the ones the tool uses. A line that does not fit in one message is blocked and nothing is written. Yolo and approvals off skip the card. Always applies to that same plan only. Hermes adds one `plugin_rule:` line to `command_allowlist` in `config.yaml`. The line stores the plan digest and does not store the photo path. The next call with the same arguments skips the card and writes the PNG. A different folder is a different line and asks again. Cron, a single-query session, an unattended platform, and plugins.isolation host stay refused even after always. Always applies only while a person is present. Anyone on your gateway can call the tool. Turn cutout off on one surface with `hermes tools disable cutout --platform cli` or `--platform telegram`. There is no platform named `gateway`. `hermes plugins remove cutout` deletes the plugin directory. `HERMES_HOME/models/cutout/u2netp.onnx` stays until you delete it. The dependency environment under `HERMES_HOME/installs` stays too. There is no cron job to remove.

The optional name is one PNG file name using letters, digits, dot, underscore, or hyphen. When name is omitted, the photo stem keeps ASCII letters, digits, dot, underscore, and hyphen. Any other run of characters becomes one hyphen. Dots, underscores, and hyphens are stripped from both ends, and the stem is cut at 40 characters. If the cleaned stem is empty, does not start with a letter or digit, contains `..`, or is not one safe PNG name, the file is `photo-cutout.png`. Otherwise the file is that stem plus `-cutout.png`. `photo.png` becomes `photo-cutout.png`. `my photo.png` becomes `my-photo-cutout.png`. Two photos whose stems clean to the same text share that name. The second call returns `exists` and does not overwrite. The call does not pick another number.

`hermes cutout call --image <photo> --folder <folder> [--name <file.png>]` prints JSON and exits 2 when nothing is written. It does not download the weights and it does not write the PNG.

### Failures

- `not_image`: Pass a photo file.
- `too_large`: Pass a smaller photo. The caps are 33554432 bytes and 24000000 pixels.
- `model_fetch`: The pinned download failed. Check the network and call again.
- `sha256_mismatch`: When `HERMES_HOME/models/cutout/u2netp.onnx` is already on disk and the hash does not match, delete that file and call again. When the download does not match, nothing is saved and there is no file to delete. Call again. The bytes were not loaded.
- `exists`: Choose a new name.
- `symlink`: The photo, the folder, the destination, or a parent directory is a symlink. Pass the real path. The reply names that directory, including letters outside ASCII. A control character, a format character, a backtick, a slash, a blank name, or a name longer than 80 characters is called that directory.
- `outside`: Pass an absolute folder and a single PNG file name.
- `hardlink`: Choose a name that is not a hard link.
- `missing`: Create the folder, or pass a photo that exists.
- `partial`: The new PNG was not confirmed. Delete that name if you do not want it, and call again.
- `approval_required`: Approve `cutout_png` in chat. The command does not write.
- `approval_denied`: Call again and approve that card.
- `approval_not_sent`: Use a shorter path.
- `approval_blocked`: plugins.isolation host refused the write.
- `unattended`: Run the tool from a chat you can answer.
- `bad_argument`: Pass image, folder, and an optional png name.
- `model_runtime`: The local model did not run. Another model is not selected.
- `dependency`: Install the plugin so its declared dependencies are present.
- `unreadable`: Check the file permissions and call again.
