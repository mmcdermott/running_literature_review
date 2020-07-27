# Research Artifacts Notes

In this document I'll summarize discrete research artifacts I've read, leaving notes.

Template
```
## [PAPER_TITLE](<LINK>)
### Summary
### Notes Section 1
...
```

## [Which Tasks Should be Learned Together in Multi-task Learning?](https://arxiv.org/pdf/1905.07553.pdf)
### Summary
This paper investigates multi-task learning in computer vision, and proposes a scheme to identify what subset of tasks should be learned together. TODO: More details

### Task-Relationships Among Multi-task Learning
They find several notable findings:
  1. More tasks = worse performance in comparison to ST models at the same (individual) capacity level, but outperform ST models that are restricted to 1/N of the capacity budget (given N tasks)
  2. They link to another paper I should read: https://arxiv.org/abs/1804.08328 for a transfer task-specific mechanism.
