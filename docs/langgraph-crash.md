# The same crash, against LangGraph

`examples/crash_and_resume.py` kills agentrt mid-charge and shows the card
charged exactly once. `examples/crash_and_resume_langgraph.py` does the same to
the LangGraph port from `examples/langgraph_port.py`, and this page records what
happened. Everything here was observed by running it, not taken from LangGraph's
documentation.

## What was tested

- langgraph 1.2.19, langgraph-checkpoint 4.3.1, langgraph-checkpoint-sqlite
  3.1.3, langchain-core 1.7.3
- Python 3.12.10 on Windows 11, SQLite through `SqliteSaver`
- The agent from `examples/langgraph_port.py`: a model node, a tools node, a
  conditional edge, agentrt's tools and wire types. Its tools node calls each
  tool directly, which is what a LangGraph tool node does. It has no
  write-ahead log, on purpose.

The kill is real. The run happens in a child process, and `charge_card` calls
`os._exit(137)` right after the charge is written and fsynced to the ledger.
No exception handler, `finally` block or background thread gets to run. A
second process then opens the same SQLite file and calls
`agent.invoke(None, config)`, LangGraph's way of resuming a thread.

The model on the resume side answers from the conversation rather than from a
fixed script: a tool result gets a final answer, anything else asks for the
charge. A fixed script has to assume where the resumed run picks up, which is
the thing being measured. An early version used one and showed three charges,
one of them caused by the script rather than LangGraph.

An in-process kill, a `BaseException` raised from the tool as the agentrt demo
does, gave the same result as `os._exit` under the default mode: two charges.

## What happened

### `durability="sync"`, five runs out of five

```
latest checkpoint: step 1, next ('tools',)
pending tasks: [('tools', None)]
write  ~__pregel_pull, model        messages
write  ~__pregel_pull, model        usage
write  ~__pregel_pull, model        branch:to:tools
```

The checkpoint after the model node is on disk and points at `tools`. The
`writes` table holds the model's output. Nothing anywhere records that the
tools task started, and the pending task shows no error. On resume LangGraph
runs `tools` again from the top:

```
charges in ledger: 2
charged 49.0
charged 49.0
```

The model was called once on resume, for the final answer. Only the tool repeated.

### `durability="async"`, the default, five runs out of five charged twice

The card was charged twice every time. Whether the model was also called
again varied between runs: in the demo's five runs it was called twice in two
runs and once in three. Over eight earlier probe runs it was three and five.

The difference lines up with how many checkpoint rows survived. In async mode
the checkpoint for a step is written in the background while the next step
runs. When the kill landed before the model step's checkpoint did, only one
row was on disk, and the resume ran the model again before running the tool
again. When it landed after, two rows were on disk and only the tool repeated.

In the probe runs `get_state` after an async crash reported step -1 or step 0
with `next=()`, which reads as nothing left to do, yet `invoke(None)` still
resumed the run and charged again. In async mode `get_state` cannot be
trusted to say whether a crashed run has work pending.

### `durability="exit"`, five runs out of five

Nothing was saved. The resume raised:

```
EmptyInputError: Received no input for __start__
```

The ledger stayed at one charge, but only because the run could not be
resumed at all. The one way left to recover is to start the run again from
its input, and doing that charged the card a second time and called the model
twice.

### agentrt, same machine, same crash

```
charges in ledger: 1
refused: tool 'charge_card' (call call_1) may have executed before the crash;
outcome unknown. Mark the tool idempotent, or resolve this run manually.
run status: needs_resolution
```

## What it means

LangGraph checkpoints what a node returned. It does not record what a node
was in the middle of doing. A tool that dies after its effect lands leaves the
database exactly as it would be if the tool had never started, so a resume
cannot tell the two apart and runs the tool again. Of the three durability
modes, `sync` gives the best result, and its best is still a repeated side effect.

This is not a bug in LangGraph. It is the checkpoint-after-the-step design that
the README describes as moving the window rather than closing it. LangGraph
leaves a crash in the middle of a side effect to the tool author. agentrt
closes the window in the runtime, by recording the intent before the call.
