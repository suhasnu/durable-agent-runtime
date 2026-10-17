# Why the loop became a graph

The first version of this agent was a `while` loop: call the model, append the
reply, run any tools it asked for, repeat until it stops asking. It is the
obvious implementation, it is correct, and `agentrt/loop.py` at commit 5 is
exactly that.

It also cannot be resumed, and not for a reason that more code would fix. The
conversation lives in a local variable called `messages`, the token total in
another called `usage`, and the position in the run is the interpreter's own
program counter. None of that is an object. When the process dies, the stack
dies, and there was never anything to write down. A forty step run that fails
at step thirty one starts again from zero, and pays again.

The same flaw shows up without a crash. When the step limit fires, the loop
raises, and the work and the token count go with it, because both are local
variables in a function that is unwinding. A tool that ran on that final step
has already had its effect, and its result is discarded unread.

So the loop's state had to stop being local variables and become an explicit,
serialisable value, and its control flow had to stop being the interpreter's
program counter and become data: a named node, with edges saying what runs
next. That is the only change in commits 7 to 9. Everything this runtime
claims afterwards, resuming a killed run, pausing for a human, replaying a
trace, forking a run at step twenty, is a consequence of that one change
rather than a feature added on top.