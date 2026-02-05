.PHONY: serve repro status

serve:
	cd book && uv run myst start

repro:
	uv run dvc repro

status:
	uv run dvc status
