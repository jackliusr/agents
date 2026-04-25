## Setup local env
* [install azure function local](https://learn.microsoft.com/en-us/azure/azure-functions/functions-run-local?tabs=linux%2Cisolated-process%2Cnode-v4%2Cpython-v2%2Chttp-trigger%2Ccontainer-apps&pivots=programming-language-python#install-the-azure-functions-core-tools)
* [install Azurite emulator for local Azure Storage development](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azurite)


## How to run

```bash
func start
```

## Testing

```bash
time curl -X POST http://localhost:7071/api/agents/Joker/run \
  --max-time 900 \
  -H "Content-Type: text/plain" \
  -d "Tell me a short joke about cloud computing."
```