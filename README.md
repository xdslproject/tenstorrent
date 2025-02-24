# xDSL Tenstorrent

An [xDSL](https://xdsl.dev) dialect and Python compiler for Tenstorrent's [Metalium](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md) framework.

Lowers the barrier to entry for programming Tensix core-enabled devices such as the Tenstorrent Grayskull and Wormhole cards.

## Development

This project is set up to work with the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/) but can be run without it.

### Some Commands

```make tests```: Runs test cases

```uv run ruff format```: Formats code

```uv run my_file.py```: Runs a Python source file using the correct `.venv`

```uv sync --extra testing```: Add testing dependencies to current `.venv`


### Project Structure

For a high-level overview of how the project is structured, test-cases can be found in `/tests`
(for the MLIR -> Metalium API) and `/examples` (for valid Python inputs). `/examples` also doubles
as a set of examples for anyone wanting to use this to implement their own Metalium programs using
this project, with those files being loaded dynamically at test time for some of the filecheck tests.
This means that changing these files also require their tests to be updated. 

The main source of this project can be found in `/tenstorrent` with `/frontend` handling the parsing
of Python input into MLIR and `/backend` containing the code that interprets the MLIR into Metalium.
`/dialects` contains the MLIR definitions for the operations and types we use, and `/tools` contains
the tools we use to handle file inputs and run them through this project. 

The Makefile defines commands that can be run with `make` and `pyproject.toml` defines some of the
project dependencies and how the project can be built for editable installation with uv. 