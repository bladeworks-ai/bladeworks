# Bladeworks

Bladeworks renders portable Final Cut Pro XML projects (`.fcpxml` files and
`.fcpxmld` bundles) to video.

## Install

```bash
python -m pip install bladeworks
```

Bladeworks requires `ffprobe` on `PATH` to inspect source media. Run the
prerequisite check after installation:

```bash
bladeworks doctor
```

## Render

```bash
bladeworks render path/to/project.fcpxmld --output output.mp4
```

See all commands and options:

```bash
bladeworks --help
```

## Test a checkout

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

## License

Bladeworks is licensed under the GNU Affero General Public License v3.0 only.
See [LICENSE](LICENSE).
