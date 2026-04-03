# Router Flowchart

```mermaid
flowchart TD
    A[START] --> B[router]
    B -->|chat| C[direct_respond]
    B -->|retrieval| D[call_retrieval_tool]
    B -->|tool| E[call_calc_or_time_tool]
    C --> F[END]
    D --> F
    E --> F
```
