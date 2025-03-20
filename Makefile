

LIT_OPTIONS ?= -v --order=lexical


.PHONY: tests
tests:
	@command uv run lit $(LIT_OPTIONS) tests/filecheck

