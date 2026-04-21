[[Activation Recomputation]]
[[Data Parallelism]]
[[Gradient accumulation]]


  

Imagine you have **one model**, but **several GPUs**.

  

Instead of giving one GPU a batch of 256 examples, you split that batch across GPUs.

  

Example with 4 GPUs:

- GPU 1 gets 64 samples
    
- GPU 2 gets 64
    
- GPU 3 gets 64
    
- GPU 4 gets 64
    

  

Each GPU has a **full copy of the model**.

  

They all do:

1. forward pass on their own mini-batch
    
2. backward pass to compute gradients for their own mini-batch
    
3. communicate with each other to **average or sum gradients**
    
4. update model parameters so all copies stay identical
    

  

So data parallelism is basically:

  

**same model, different slices of data, then synchronize gradients**

  

That’s the core.

---

## **Why do we need all that other stuff?**

  

Because training at scale is a three-headed monster:

- **memory**
    
- **communication**
    
- **throughput**
    

  

The techniques you asked about are mostly tricks for wrestling one of those heads.

---

# **1. Activation recomputation**

  

Also called **activation checkpointing**.

  

### **What problem does it solve?**

  

During forward pass, the model creates many intermediate values called **activations**.

  

Backward pass needs them to compute gradients.

  

So the naïve strategy is:

- save lots of activations during forward
    
- use them during backward
    

  

This is fast, but eats a huge amount of GPU memory.

  

### **What recomputation does**

  

Instead of saving everything, you save only some checkpoints.

  

Later during backward, if you need missing activations, you **recompute** them by running part of the forward again.

  

So you trade:

- **less memory**
    
- for **more compute**
    

  

### **Plain picture**

  

Without recomputation:

- forward: save every breadcrumb
    
- backward: follow breadcrumbs
    

  

With recomputation:

- forward: save only a few landmarks
    
- backward: walk the path again when needed
    

  

### **Why it matters**

  

It lets you train:

- larger models
    
- longer sequences
    
- bigger micro-batches
    

  

on the same GPU memory budget.

  

### **Cost**

  

You do extra computation, so training becomes slower.

---

# **2. Gradient accumulation**

  

### **What problem does it solve?**

  

Sometimes your desired batch is too big to fit in memory.

  

Suppose you want effective batch size = 256, but one GPU can only hold 32 samples at once.

  

Then you can do 8 smaller steps:

- run batch 32
    
- compute gradients
    
- **do not update parameters yet**
    
- run another batch 32
    
- add gradients
    
- repeat 8 times
    
- then do one optimizer step
    

  

So gradient accumulation means:

  

**pretend several small batches are one larger batch by summing gradients before updating**

  

### **Important distinction**

  

There are two batch sizes now:

- **micro-batch**: what fits in memory for one forward/backward
    
- **effective batch**: micro-batch × accumulation steps × number of GPUs
    

  

Example:

- 4 GPUs
    
- micro-batch = 8 per GPU
    
- accumulation steps = 4
    

  

Effective batch = 4 × 8 × 4 = 128

  

### **Why it matters**

  

It helps when:

- memory is too small
    
- you want a larger effective batch for optimization stability
    

  

### **Cost**

  

You get less frequent optimizer updates, and each update takes longer wall-clock time.

---

# **3. Overlap gradient communication**

  

This is about **hiding communication time**.

  

### **The basic problem**

  

In data parallelism, after backward pass, GPUs need to exchange gradients.

  

That communication can be slow.

  

If you do it the dumb way:

1. compute all gradients
    
2. stop
    
3. communicate all gradients
    
4. continue
    

  

then GPUs spend time waiting around like workers staring at a jammed conveyor belt.

  

### **Overlap idea**

  

Backward pass computes gradients layer by layer, usually from top layers to bottom layers.

  

As soon as gradients for one group of parameters are ready, you can start communicating them **while backward is still computing gradients for lower layers**.

  

So communication and computation happen at the same time.

  

### **Plain picture**

  

Instead of:

- compute everything
    
- then communicate
    

  

you do:

- compute chunk A gradients → start sending
    
- while sending A, compute chunk B
    
- while sending B, compute chunk C
    

  

This is called overlapping communication with backward computation.

  

### **Why it matters**

  

It reduces idle time and improves scaling across GPUs.

---

# **4. Gradient bucketing**

  

This is closely related.

  

### **What problem does it solve?**

  

Models have many parameter tensors.

  

If you communicate every tiny gradient tensor separately, you get:

- too many small communication calls
    
- bad bandwidth usage
    
- large overhead
    

  

That’s like mailing 50,000 tiny envelopes instead of a few boxes.

  

### **Bucketing idea**

  

Group multiple gradient tensors into a larger **bucket**.

  

When a bucket is full, communicate the whole bucket together.

  

So bucketing means:

  

**bundle many small gradients into bigger chunks for more efficient communication**

  

### **Why it helps**

  

Networks like larger messages better than endless tiny ones.

  

It also helps overlap, because the framework can say:

- this bucket is ready
    
- launch all-reduce now
    

  

instead of waiting on every single parameter one by one.

---

# **5. Interplay with gradient accumulation**

  

This is the part that gets slippery.

  

## **First principle**

  

During gradient accumulation, you are **not updating parameters every backward pass**.

  

You do several micro-steps before one optimizer step.

  

That changes how communication should be handled.

---

## **Case A: naïve accumulation with full sync every micro-step**

  

Suppose you have 8 accumulation steps.

  

Each micro-step does:

1. forward
    
2. backward
    
3. all-reduce gradients across GPUs
    
4. add into accumulated gradients
    
5. no optimizer step yet
    

  

After 8 micro-steps:

- optimizer step
    

  

This works, but communication happens every micro-step.

  

That can be expensive.

  

So even though you delayed the optimizer step, you did **not** delay communication.

---

## **Case B: defer synchronization during accumulation**

  

A more efficient pattern is:

- for micro-steps 1 to 7:
    
    - forward
        
    - backward
        
    - accumulate gradients locally
        
    - **do not synchronize across GPUs yet**
        
    
- on micro-step 8:
    
    - backward
        
    - synchronize gradients
        
    - optimizer step
        
    

  

This avoids repeated communication.

  

In PyTorch DDP, this is often done with no_sync() for non-final accumulation steps.

  

### **Why this helps**

  

Communication cost drops a lot, especially when accumulation steps are large.

  

### **Tradeoff**

  

You lose some overlap opportunities on skipped sync steps, and the final sync may be bigger or more bursty.

  

But usually it is still worth it.

---

## **How bucketing interacts with accumulation**

  

Bucketing only matters when you are actually communicating gradients.

  

So:

- if you sync every micro-step, bucketing happens every micro-step
    
- if you defer sync until the last micro-step, bucketing mostly matters on that final synchronized backward
    

  

In other words, **accumulation changes how often buckets get communicated**

  

not what a bucket is.

---

## **How overlap interacts with accumulation**

  

If you sync gradients every micro-step:

- overlap can happen every backward pass
    

  

If you skip sync for most micro-steps:

- there is no communication to overlap on those steps
    
- only the final synchronized backward gets overlap
    

  

So gradient accumulation often reduces communication frequency, but also reduces chances to hide communication during the skipped steps.

  

Still, in practice, avoiding communication usually saves more than overlap would.

---

## **How activation recomputation interacts with accumulation**

  

These two solve different problems:

- activation recomputation helps **memory**
    
- gradient accumulation also helps **memory**, by letting you use smaller micro-batches
    

  

Together they are often used to squeeze training into limited memory.

  

Example:

- recomputation frees memory inside each micro-batch
    
- accumulation lets many micro-batches behave like one large batch
    

  

### **Tradeoff stack**

  

If you use both heavily:

- memory usage goes down
    
- compute cost goes up
    
- time per optimizer step can increase a lot
    

  

You are basically buying memory with extra work.

---

# **Mental model: one training step**

  

Let’s say:

- 4 GPUs
    
- micro-batch = 8
    
- accumulation steps = 4
    
- DDP enabled
    
- recomputation enabled
    
- bucketing enabled
    

  

Then one optimizer step looks like this:

  

## **Micro-step 1**

- each GPU runs forward on 8 samples
    
- some activations are not stored, only checkpointed
    
- backward recomputes missing activations
    
- gradients are computed
    
- maybe gradients are kept local if using no-sync accumulation
    

  

## **Micro-step 2**

- same thing
    
- gradients add to existing gradient buffers
    

  

## **Micro-step 3**

- same
    

  

## **Micro-step 4**

- forward
    
- backward
    
- as gradients for parameter buckets become ready, buckets are all-reduced across GPUs
    
- communication may overlap with remaining backward computation
    
- final accumulated gradients now represent full effective batch
    
- optimizer step happens
    
- gradients zeroed
    

  

Effective batch size here is:

  

4 GPUs × 8 micro-batch × 4 accumulation = 128

---

# **Quick intuition for each technique**

  

## **Data parallelism**

  

Use more GPUs by splitting data.

  

## **Activation recomputation**

  

Save less memory by redoing part of forward later.

  

## **Gradient accumulation**

  

Use many small batches to imitate one large batch.

  

## **Overlap gradient communication**

  

Hide network time under backward compute.

  

## **Gradient bucketing**

  

Combine many small gradient messages into fewer large ones.

---

# **Simple analogy**

  

Imagine a bakery with 4 kitchens.

  

## **Data parallelism**

  

Each kitchen bakes the same recipe for different trays of cookies.

  

## **Gradient accumulation**

  

Instead of sending the head chef feedback after every tiny tray, each kitchen waits until several trays are done, then combines the notes.

  

## **Activation recomputation**

  

Instead of storing every intermediate prep bowl, the kitchen re-mixes some ingredients later to save counter space.

  

## **Gradient bucketing**

  

Instead of sending one text message per cookie, the kitchen sends one summary per tray.

  

## **Overlap gradient communication**

  

While one tray is still baking, the kitchen already sends feedback about the previous tray.

---

# **One subtle but important point**

  

Gradient accumulation is **not the same** as simply increasing batch size in all situations.

  

In theory, if everything is scaled correctly, it approximates a larger batch.

  

In practice, differences can appear because of:

- dropout randomness
    
- batchnorm behavior
    
- optimizer details
    
- gradient clipping timing
    
- mixed precision scaling
    
- scheduler stepping conventions
    

  

So it is “effectively larger batch,” but not always perfectly identical.

---

# **Rule of thumb for when to use what**

  

Use **data parallelism** when one model copy fits on each GPU and you want more throughput.

  

Use **activation recomputation** when model/sequence/micro-batch does not fit in memory.

  

Use **gradient accumulation** when micro-batch is too small due to memory, but you still want a larger effective batch.

  

Use **bucketing** when doing distributed gradient communication, which is almost always.

  

Use **overlap** when communication is becoming a scaling bottleneck.

---

# **The shortest practical summary**

  

A compact equation of pain:

  

**memory pressure**

→ activation recomputation + smaller micro-batches + gradient accumulation

  

**communication pressure**

→ bucketing + overlap + maybe defer sync during accumulation

  

**throughput scaling**

→ good balance among compute, memory, and communication

  

If you want, next I can draw a **timeline diagram of backward pass + all-reduce + accumulation** and explain exactly what DDP is doing under the hood.