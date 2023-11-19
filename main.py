import os

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# from gpt4all import GPT4All
#
# model = GPT4All(
#     model_name="mistral-7b-instruct-v0.1.Q4_0.gguf", device="gpu", n_threads=14
# )
# with model.chat_session():
#     while True:
#         try:
#             user_input = input("enter your question:\n")
#             response1 = model.generate(prompt=user_input.strip(), temp=0)
#             print(response1)
#         except KeyboardInterrupt:
#             print(model.current_chat_session)
#             exit()

local_path = f"{os.environ['HOME']}/.cache/gpt4all/mistral-7b-instruct-v0.1.Q4_0.gguf"
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

SYSTEM_PROMPT = "### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n"

# PROMPT_TEMPLATE = """
# ### User:
# Your goal is to provide a summary of the task done by analyzing a git commit and diff.
# Consider the git commit message but also the code changes in the diff to interpret what was achieved in this task.
# Your output will be presented in a meeting and should be a concise description of how the code changed.
# Each task should me mentioned only once.
#
# Git commit:
# {commit}
#
# Git diff:
# {diff}
#
# ### Response:\n
# """

PROMPT_TEMPLATE = """
### System:
Generate a concise task description in past tense based on the git commit message and git diff provided.
Your response should state clearly in a single line what improvements were made to the code.
The git diff provides all changes made to the code, therefore you can use it to interpret how the code changed.

### User:
Git message:
{message}

Git diff:
{diff}

### Response:\n
"""

prompt = PromptTemplate(
    template=f"{SYSTEM_PROMPT}{PROMPT_TEMPLATE}", input_variables=["commits"]
)

llm = GPT4All(
    model=local_path,
    callback_manager=callback_manager,
    verbose=True,
    # device="nvidia",
    n_threads=14,
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

commit = """
4502bb50d044116d9a56e51e3130ea5a8d9b0258 (HEAD -> main, origin/main, origin/HEAD) improving tree printing
"""

diff = """
index 653b298..843c146 100644
--- a/graph/adjacency.go
+++ b/graph/adjacency.go
@@ -1,6 +1,7 @@
 package graph
 
 import (
+	"container/heap"
 	"fmt"
 	"main/list"
 	"math"
@@ -129,34 +130,37 @@ func DFS(graph AdjacencyList, source int, needle int) ([]int, error) {
 	return outPath, nil
 }
 
-func hasUnvisited(seen []bool, dists []float64) bool {
-	for i := 0; i < len(seen); i++ {
-		if !seen[i] && dists[i] < infinity { // TODO infinity
-			return true
-		}
-	}
-	return false
+type nodeDist struct {
+	node int
+	dist float64
 }
 
-func getLowestUnvisited(seen []bool, dists []float64) int {
-	idx := -1
-	ld := infinity
+type DistHeap []nodeDist
 
-	for i := 0; i < len(seen); i++ {
-		if seen[i] {
-			continue
-		}
-		if ld > dists[i] {
-			ld = dists[i]
-			idx = i
-		}
-	}
-	return idx
+func (h DistHeap) Len() int           { return len(h) }
+func (h DistHeap) Less(i, j int) bool { return h[i].dist < h[j].dist }
+func (h DistHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
+
+func (h *DistHeap) Push(x any) {
+	// Push and Pop use pointer receivers because they modify the slice's length,
+	// not just its contents.
+	*h = append(*h, x.(nodeDist))
+}
+
+func (h *DistHeap) Pop() any {
+	old := *h
+	n := len(old)
+	x := old[n-1]
+	*h = old[0 : n-1]
+	return x
 }
 
 func DijkstraList(graph AdjacencyList, source int, sink int) []int {
 	seen := make([]bool, len(graph))
 	dists := make([]float64, len(graph))
+	h := &DistHeap{}
+	heap.Init(h)
+	heap.Push(h, nodeDist{node: source, dist: 0})
 	for i := range dists {
 		dists[i] = infinity
 	}
@@ -166,31 +170,31 @@ func DijkstraList(graph AdjacencyList, source int, sink int) []int {
 	}
 	dists[source] = 0
 
-	var curr int
+	var nd nodeDist
 	var dist float64
 	var adjs []GraphEdge
 	var edge GraphEdge
-	// TODO use heap to improve runtime
-	for hasUnvisited(seen, dists) {
-		curr = getLowestUnvisited(seen, dists)
-		seen[curr] = true
+	for h.Len() > 0 {
+		nd = heap.Pop(h).(nodeDist)
+		seen[nd.node] = true
 
-		adjs = graph[curr]
+		adjs = graph[nd.node]
 		for i := 0; i < len(adjs); i++ {
 			edge = adjs[i]
 			if seen[edge.to] {
 				continue
 			}
 
-			dist = dists[curr] + edge.weight
+			dist = nd.dist + edge.weight
 			if dist < dists[edge.to] {
 				dists[edge.to] = dist
-				prev[edge.to] = curr
+				heap.Push(h, nodeDist{node: edge.to, dist: dist})
+				prev[edge.to] = nd.node
 			}
 		}
 	}
 	out := make([]int, 0)
-	curr = sink
+	curr := sink
 	for prev[curr] != -1 {
 		out = append(out, curr)
 		curr = prev[curr]
"""

message = "improving Dijkstra running time"

print(prompt.format(diff=diff, message=message))

llm_chain.run(diff=diff, message=message)
print()
