

LIT_OPTIONS ?= -v --order=smart


.PHONY: tests
tests:
	@command uv run lit $(LIT_OPTIONS) tests/filecheck

