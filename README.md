# xDSL Tenstorrent

An [xDSL](https://xdsl.dev) dialect and Python compiler for Tenstorrent's [Metalium](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md) framework.

Lowers the barrier to entry for programming Tensix core-enabled devices such as the Tenstorrent Grayskull and Wormhole cards.

### Development

```make tests```: Runs test cases

```uv run ruff format```: Formats code

```uv sync --extra testing```: Add testing dependencies to current `.venv`
