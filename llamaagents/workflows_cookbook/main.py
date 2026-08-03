#!/usr/bin/env python
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)
import random
from llama_index.utils.workflow import (
    draw_all_possible_flows,
    draw_most_recent_execution,
)
from llama_index.llms.ollama import Ollama


from llama_index.llms.openai import OpenAI
from llama_index.core.workflow.errors import WorkflowTimeoutError
import asyncio

class OllamaGenerator(Workflow):
    @step
    async def generate(self, ev: StartEvent) -> StopEvent:
        llm = Ollama(model="qwen3.6:35b-a3b-q4_K_M", request_timeout=300.0)
        response = await llm.acomplete(ev.query)
        return StopEvent(result=str(response))


draw_all_possible_flows(OllamaGenerator, filename="trivial_workflow.html")

class FailedEvent(Event):
    error: str


class QueryEvent(Event):
    query: str


class LoopExampleFlow(Workflow):
    @step
    async def answer_query(
        self, ev: StartEvent | QueryEvent
    ) -> FailedEvent | StopEvent:
        query = ev.query
        # try to answer the query
        random_number = random.randint(0, 1)
        if random_number == 0:
            return FailedEvent(error="Failed to answer the query.")
        else:
            return StopEvent(result="The answer to your query")

    @step
    async def improve_query(self, ev: FailedEvent) -> QueryEvent | StopEvent:
        # improve the query or decide it can't be fixed
        random_number = random.randint(0, 1)
        if random_number == 0:
            return QueryEvent(query="Here's a better query.")
        else:
            return StopEvent(result="Your query can't be fixed.")

draw_all_possible_flows(LoopExampleFlow, filename="loop_workflow.html")

class GlobalExampleFlow(Workflow):
    @step
    async def setup(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        # load our data here
        await ctx.store.set("some_database", ["value1", "value2", "value3"])

        return QueryEvent(query=ev.query)

    @step
    async def query(self, ctx: Context, ev: QueryEvent) -> StopEvent:
        # use our data with our query
        data = await ctx.store.get("some_database")

        result = f"The answer to your query is {data[1]}"
        return StopEvent(result=result)

draw_all_possible_flows(GlobalExampleFlow, filename="global_workflow.html")

class WaitExampleFlow(Workflow):
    @step
    async def setup(self, ctx: Context, ev: StartEvent) -> StopEvent:
        if hasattr(ev, "data"):
            await ctx.store.set("data", ev.data)

        return StopEvent(result=None)

    @step
    async def query(self, ctx: Context, ev: StartEvent) -> StopEvent:
        if hasattr(ev, "query"):
            # do we have any data?
            if hasattr(self, "data"):
                data = await ctx.store.get("data")
                return StopEvent(result=f"Got the data {data}")
            else:
                # there's non data yet
                return None
        else:
            # this isn't a query
            return None

draw_all_possible_flows(WaitExampleFlow, filename="wait_workflow.html")


class InputEvent(Event):
    input: str


class SetupEvent(Event):
    error: bool


# class QueryEvent(Event):
#     query: str


class CollectExampleFlow(Workflow):
    @step
    async def setup(self, ctx: Context, ev: StartEvent) -> SetupEvent:
        # generically start everything up
        if not hasattr(self, "setup") or not self.setup:
            self.setup = True
            print("I got set up")
        return SetupEvent(error=False)

    @step
    async def collect_input(self, ev: StartEvent) -> InputEvent:
        if hasattr(ev, "input"):
            # perhaps validate the input
            print("I got some input")
            return InputEvent(input=ev.input)

    @step
    async def parse_query(self, ev: StartEvent) -> QueryEvent:
        if hasattr(ev, "query"):
            # parse the query in some way
            print("I got a query")
            return QueryEvent(query=ev.query)

    @step
    async def run_query(
        self, ctx: Context, ev: InputEvent | SetupEvent | QueryEvent
    ) -> StopEvent | None:
        ready = ctx.collect_events(ev, [QueryEvent, InputEvent, SetupEvent])
        if ready is None:
            print("Not enough events yet")
            return None

        # run the query
        print("Now I have all the events")
        print(ready)

        result = f"Ran query '{ready[0].query}' on input '{ready[1].input}'"
        return StopEvent(result=result)

draw_all_possible_flows(CollectExampleFlow, "collect_workflow.html")
    

async def run_workflow(
    workflow_cls,
    timeout: float = 120,
    verbose: bool = False,
    max_retries: int = 3,
    **kwargs,
):
    """Run a workflow, retrying on the intermittent WorkflowTimeoutError.

    The underlying workflows runtime occasionally fails to dispatch an emitted
    event to the next step ("No steps active"), which surfaces as a rare
    WorkflowTimeoutError. Re-running with a fresh instance succeeds reliably,
    so we retry a few times before giving up.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        wf = workflow_cls(timeout=timeout, verbose=verbose)
        try:
            return await wf.run(**kwargs)
        except WorkflowTimeoutError as exc:
            last_exc = exc
            print(
                f"  [retry {attempt}/{max_retries}] "
                f"{workflow_cls.__name__} timed out; re-running..."
            )
    assert last_exc is not None
    raise last_exc


async def main():
    # Trivial workflow: Ollama generation
    print("Trivial workflow")
    result = await run_workflow(
        OllamaGenerator, query="What's LlamaIndex?"
    )
    print(result)

    # Loop workflow: retry/improve query
    print("Loop workflow")
    result = await run_workflow(
        LoopExampleFlow, verbose=True, query="What's LlamaIndex?"
    )
    print(result)

    print("Global workflow")
    result = await run_workflow(
        GlobalExampleFlow, verbose=True, query="What's LlamaIndex?"
    )
    print(result)

    # Wait workflow: context/state examples
    print("Wait workflow")
    result = await run_workflow(
        WaitExampleFlow, verbose=True, query="Can I kick it?"
    )
    if result is None:
        print("No you can't")
    print("---")
    result = await run_workflow(WaitExampleFlow, data="Yes you can")
    print("---")
    result = await run_workflow(
        WaitExampleFlow, verbose=True, query="Can I kick it?"
    )
    print(result)

    print("Collect workflow")
    result = await run_workflow(
        CollectExampleFlow,
        input="Here's some input",
        query="Here's my question",
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())