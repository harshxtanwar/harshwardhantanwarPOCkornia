# Proof of Concept for PyG - Transpiler Project for Ivy

1. [My understanding with Ivy's transpiler](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#my-understanding-with-ivys-transpiler)
   - [Transpiling functions eagerly](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#1-transpiling-function-eagerly)
   - [Ivy transpiling Functions, Models and Frameworks !](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#2-ivy-transpiling-functions-models-and-frameworks-)
2. [Objective of the project](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#objective-of-the-project)
   - [Main Objective](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#1-main-objective)
   - [Workflow and how exactly I will proceed with the project](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#2-workflow-and-how-exactly-i-will-proceed-with-the-project)
3. [Listing Functions !](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#listing-functions-)
   - [An example of list of functions used in torch_geometric.nn](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#1-an-example-of-list-of-functions-used-in-torch_geometricnn-file-directory-in-pyg-repository)
   - [An example of list of functions used in torch_geometric.data](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#2-an-example-of-list-of-functions-used-in-torch_geometricdata-file-directory-in-pyg-repository)
   - [An example of list of functions used in torch_geometric.loader](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#3-an-example-of-list-of-functions-used-in-torch_geometricloader-file-directory-in-pyg-repository)
   - [An example of list of functions used in torch_geometric.sampler](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#4-an-example-of-list-of-functions-used-in-torch_geometricsampler-file-directory-in-pyg-repository)
4. [Implementation Example, Missing Torch Frontend](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#implementation-example-missing-torch-frontend)
   - [Pytorch Geometric's Code](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#1-pytorch-geometrics-code)
   - [Ivy's existing code for torch.bincount](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#2-ivys-existing-code-for-torchbincount)
   - [Solution](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#3-solution---implementing-torchs-frontend-for-torchbincount)

## My understanding with Ivy's transpiler
In this section, I aim to convey to you my understanding of Ivy's transpiler. I will talk about how
Ivy transpiles functions, modules and frameworks eagerly and lazily !

### 1. Transpiling function eagerly
- step 1: A graph is compiled for the function that Ivy is trying to transpile in the framework that the function is in.  
          For example: in case of a PyG function, we will get a Torch graph.  
- step 2: Each of the nodes in the compiled graph are replaces with the Ivy frontend functions of the corresponding framework  
          For example: in the case of a Torch graph, all of the functions will get replaced by ivy.functional.frontends.torch functions.  
- step 3: All of the arrays in arguments and keyword arguments are converted to the targeted framework.
          For example: if the to argument is given as `to=numpy`, then the arrays of all arguments are converted to numpy arrays.
- step 4: A new optimized graph is compiled using the graph compiled after step 2 and the args and kwargs after step 3 with removed functional wrappings
          For example: In our case, the torch graph will get replaced with numpy functions, and the args and kwargs are already converted to numpy arrays.
          
> Note: that the function demonstrated below is not the real transpiler function of ivy, it is just made to show my basic understanding on how things work 
> with the transpiler
```
def _transpile_function_eagerly(
  fn,
  to: str,
  args: Tuple,
  kwargs: dict,
):
 
"""
fn
  The function to transpile.
to
  The framework to transpile to.
args
  The positional arguments to pass
  to the function for tracing.
kwargs
  The keyword arguments to pass
  to the function for tracing.
"""
  
  # step 1
  source_graph = \
    ivy.compile_graph(fn, *args, **kwargs)
  
  # step 2
  frontend_graph = \
    _replace_nodes_with_frontend_functions(source_graph)
  
  # step 3
  args, kwargs = _src_to_trgt_arrays(args, kwargs)
  
  # step 4 
  target_graph = \
    ivy. compile_graph(frontend_graph, *args, **kwargs, to=to)

  return target_graph

```
### 2. Ivy transpiling Functions, Models and Frameworks !

In reality, the Ivy's transpiler function is quite flexible, and is able to transpile Functions, Models and Frameworks into a  
framework of your choice. 

- Ivy can Transpile either functions, trainable models or importable python modules,
with any number of combinations as many number of times !

- If no “objs” are provided, the function returns a new transpilation function
which receives only one object as input, making it usable as a decorator.

- If neither “args” nor “kwargs” are specified, then the transpilation will occur
lazily, upon the first call of the transpiled function as the transpiler need some arguments
in order to trace all of the functions required to compile a graph, otherwise transpilation is eager.

```
def transpile(
   *objs,
   to: Optional[str] = None,
   args: Optional[Tuple] = None,
   kwargs: Optional[dict] = None,
):
"""
objs
   The functions, models or modules
   to transpile.
to
   The framework to transpile to.
args
   The positional arguments to pass
   to the function for tracing.
kwargs
   The keyword arguments to pass
   to the function for tracing.
"""

# The code below this is made private by ivy, but from the example above for eager transpilation, I covered 
# all of the steps that happen when a user trys to transpile anythinf from Ivy, and not just a function.

```
## Objective of the project
In this section, I aim to convey to you as to what exactly the projects aims to achieve after it's completion and what all things
are exactly required to be done in order to complete this project.

### 1. Main Objective
- The main aim of this project is to make **Pytorch Geometric (PyG)** compatible with all other machine learning frameworks supported by Ivy  
like Numpy, Jax, Paddle and Tensorflow.
- After the successful implementation of this project, users of PyG will be able to transpile the whole 
PyG module to a framework of their choice !
- The users will highly benefit from this as the runtime efficiency is greatly improved when using a JAX backend on TPU, compared  
to the original PyTorch implementation.

### 2. Workflow and how exactly I will proceed with the project
Below are the steps I will follow to work on the project in a chronological order.
- Finding and creating a list of all of the pyTorch function used in the PyG's Module.
- Eliminating the functions from the list which already have been implemented in Ivy's both backend and frontend
- Creating a list of functions that already have been implemented in the backend but don't have any frontend wrapper around them
and creating another list of functions that need to be implemented for all backends in Ivy along with a frontend wrapper.
- You can find a list of such function the section below in my proof of concept.
- After finalising the list, I will start working with multiple pull requests to first finish writing codes for the functions with **missing frontend wrappers** in Ivy's repository.
- Then I will work on multiple pull requests to implement functions that have **both missing backend handlers and frontend wrapper along with the test cases** in Ivy's repository.
- Once this is done, I will make a pull request to PyG's repository where I will implement PyG's framework handlers to enable transpilation to provide PyG's users with functions that can transpile PyG to a framework of their choice.
- I will create Google Colab Demos showcasing how GNNs built using PyG can be used in TensorFlow and JAX projects (or any other framework) for the users of PyG.
- I will also create Google Colab Demos showing how the runtime efficiency is greatly improved when using a JAX backend on TPU, compared to the original PyTorch implementation.

 ## Listing Functions !
 A list of all torch functions used in PyG is to be made, particulary in  
 
> I have listed down all of the torch functions used in torch_geometric.nn, torch_geometric.data, torch_geometric.loader and torch_geometric.sampler below.
> this is just a list of all types of functions/modules of the torch used in all of the files combined in the directory. I will further show examples of functions with missing backends and frontends in the the below section 

- torch_geometric
- torch_geometric.nn
- torch_geometric.data
- torch_geometric.loader
- torch_geometric.sampler
- torch_geometric.datasets
- torch_geometric.transforms
- torch_geometric.utils
- torch_geometric.explain
- torch_geometric.contrib
- torch_geometric.graphgym
- torch_geometric.profile

### 1. An example of list of functions used in torch_geometric.nn file directory in PyG repository  
 
> (PS: I had to browse through **167 files** inside of the torch_geometric.nn directory in PyG's repository to make a whole list of these functions)

[
torch.nn.parameter , torch.long, torch.no_grad, torch.nn.ModuleList, torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Tanh, torch.cat, 
torch.nn.softplus, torch.nn.Sigmoid, torch.zeros_like, torch.autograd.grad, torch.zeros, torch.enable_grad, torch.nn.GRU, torch.nn.LSTM, torch.nn.MultiheadAttention, torch.stack, torch.mean, torch.bincount, torch.cumsum, torch.sort, torch.arange, torch.float, torch.log, torch.arange, torch.nn.LayerNorm, torch.nn.MultiheadAttention, torch.nn.init.calculate_gain, torch.nn.init.xavier_normal_ torch.nn.functional.normalize, torch.eye, torch.nn.init.kaiming_uniform_, torch.nn.functional.dropout, torch.nn.ReLu, torch.nn.parameter.UninitializedParameter, torch.nn.BatchNorm1d, torch.sparse_csc, torch.matmul, torch.clamp, torch.sqrt, torch.full, torch.nn.functional.gelu, torch.nn.functional.softmax, torch.nn.functional.leaky_relu, torch.nn.GRUCell, torch.nn.Param, torch.jit._overload, torch.sparse_csc, torch.ones, torch.addmm, torch.nn.BatchNorm1d, torch.nn.InstanceNorm1d, torch.nn.Sequential, torch.nn.Identity, torch.nn.MultiheadAttention, torch.exp, torch.tanh, torch.sum, torch.nn.Embedding, torch.nn.functional.gelu, torch.utils.hooks.RemovableHandle, torch.uint8, torch.int8, torch.int32, torch.int64, torch.sparse_coo, torch.sparse_csr, torch.sparse_csc, torch.jit.unused, torch.utils.data.Dataloader, torch.bincount, torch.atan2, torch.cross, torch.index_select, torch.einsum, torch.bmm, torch.ones_like, torch.mul, torch.ones_like, torch.where, torch.is_floating_point, torch.gather, torch.nn.functional.normalize, torch.no_grad, torch.nn.ELU, torch.nn.Conv1d, torch.norm, torch.nn.init.uniform_, torch.onnx.is_in_onnx_export, torch.nn.utils.rnn.pad_sequence, torch.unique, torch.randint, torch.zeros_like, torch.nn.functional.binary_cross_entropy_with_logits, torch.nn.functional.margin_ranking_loss, torch.nn.init.xavier_uniform_, torch.nn.init.uniform_, torch.linalg.vector_norm, torch.cos, torch.sin, torch.sigmoid, torch.randn_like, torch.mean, torch.device, torch.bool, torch.utils.checkpoint.checkpoint, torch.atan2, torch.from_numpy, torch.linspace, torch.nn.modules.loss._Loss, torch.nn.BCEWithLogitsLoss, torch.nn.functional.logsigmoid, torch.clone, torch.rand, torch.randint, torch.ops.torch_cluster.random_walk, torch.jit.export, torch.jit._overload_method, torch.set_grad_enabled, torch.chunk, torch.pow, torch.load, torch.linspace, torch.nn.functional.nll_loss, torch.randperm, torch.log_softmax, torch.empty, torch.nn.functional.batch_norm, torch.cdist, torch.nn.functional.instance_norm, torch.nn.functional.layer_norm, torch.jit.unused, torch.LongTensor, torch.div, torch.empty_like, torch.nn.KLDivLoss, torch.min, torch.is_tensor, torch.logspace, torch.nn.ModulDict, torch.fx.Graph, torch.fx.GraphModule, torch.fx.Node, torch.fx.map_arg, torch.nn.init.orthogonal_, torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR, torch.save, torch.optim.lr_scheduler.ReduceLROnPlateau, torch.optim.lr_scheduler, torch.jit.ScriptModule, torch.max 
]

### 2. An example of list of functions used in torch_geometric.data file directory in PyG repository
> (PS: I had to browse through **19 files** inside of the torch_geometric.data directory in PyG's repository to make a whole list of these functions)

[
torch.utils.data.DataLoader, torch.utils.data.get_worker_info, torch.cat.torch.tensor, torch.arange, torch.device, torch.full, torch.cumsum, torch.cat, torch.stack, torch.cuda.Stream, torch.arange, torch.unique, torch.empty_like, torch.utils.data.IterDataPipe
, torch.utils.data.functional_datapipe, torch.utils.data.datapipes.iter.Batcher, torch.utils.data.datapipes.iter.IterBatcher, torch.float, torch.utils.data.Dataset, torch.is_floating_point, torch.load, torch.from_numpy, torch.randn, torch.full, torch.long, torch.equal, torch.nn.utils.rnn.PackedSequence
]

### 3. An example of list of functions used in torch_geometric.loader file directory in PyG repository
> (PS: I had to browse through **20 files** inside of the torch_geometric.loader directory in PyG's repository to make a whole list of these functions)

[
torch.utils.data.dataloader._BaseDataLoaderIter, torch.load, torch.arange, torch.save, torch.stack, torch.utils.data.DataLoader, torch.cat, torch.arange, torch.stack, torch.utils.data.dataloader.default_collate, torch.float, torch.randperm, torch.long, torch.zeros, torch.empty_like, torch.isnan, torch.randint, torch.rand, torch.randint, torch.utils.data.WeightedRandomSampler, torch.get_num_threads, torch.set_num_threads, torch.ops.torch_sparse.ego_k_hop_sample_adj, torch.bool, torch.ops.torch_sparse.ptr2ind, torch.utils.data.get_worker_info, torch.index_select, torch.from_numpy, torch.int64
]

### 4. An example of list of functions used in torch_geometric.sampler file directory in PyG repository
> (PS: I had to browse through **4 files** inside of the torch_geometric.sampler directory in PyG's repository to make a whole list of these functions)

[
torch.Tensor, torch.long, torch.randint, torch.multinomial, torch.ops.torch_sparse.hgt_sample, torch.int64, torch.ops.pyg.hetero_neighbor_sample, torch.ops.pyg.neighbor_sample, torch.ops.torch_sparse.neighbor_sample, torch.ones, torch.cat, torch.arange, torch.stack, torch.from_numpy, torch.empty, torch.device, torch.zeros
]


## Implementation Example, Missing Torch Frontend

torch.bincount function is one such function which is used in Pyg, in the 
pytorch_geometric/torch_geometric/nn/aggr/quantile.py directory and you can view the use of torch.bincount
function in this directory at this [link](https://github.com/pyg-team/pytorch_geometric/blob/7469edee6edae1afd8a9dc61b1494ec6412195aa/torch_geometric/nn/aggr/quantile.py#L78)


### 1. Pytorch Geometric's Code
View the link above to view the exact location of PyG's functio and torch.bincount in their repository in github

```
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        dim = x.dim() + dim if dim < 0 else dim

        self.assert_index_present(index)
        assert index is not None  # Required for TorchScript.

        count = torch.bincount(index, minlength=dim_size or 0)
        cumsum = torch.cumsum(count, dim=0) - count

        q_point = self.q * (count - 1) + cumsum
        q_point = q_point.t().reshape(-1)

        shape = [1] * x.dim()
        shape[dim] = -1
        index = index.view(shape).expand_as(x)

        # Two sorts: the first one on the value,
        # the second (stable) on the indices:
        x, x_perm = torch.sort(x, dim=dim)
        index = index.take_along_dim(x_perm, dim=dim)
        index, index_perm = torch.sort(index, dim=dim, stable=True)
        x = x.take_along_dim(index_perm, dim=dim)

        # Compute the quantile interpolations:
        if self.interpolation == 'lower':
            quantile = x.index_select(dim, q_point.floor().long())
        elif self.interpolation == 'higher':
            quantile = x.index_select(dim, q_point.ceil().long())
        elif self.interpolation == 'nearest':
            quantile = x.index_select(dim, q_point.round().long())
        else:
            l_quant = x.index_select(dim, q_point.floor().long())
            r_quant = x.index_select(dim, q_point.ceil().long())

            if self.interpolation == 'linear':
                q_frac = q_point.frac().view(shape)
                quantile = l_quant + (r_quant - l_quant) * q_frac
            else:  # 'midpoint'
                quantile = 0.5 * l_quant + 0.5 * r_quant

        # If the number of elements is zero, fill with pre-defined value:
        mask = (count == 0).repeat_interleave(self.q.numel()).view(shape)
        out = quantile.masked_fill(mask, self.fill_value)

        if self.q.numel() > 1:
            shape = list(out.shape)
            shape = (shape[:dim] + [shape[dim] // self.q.numel(), -1] +
                     shape[dim + 2:])
            out = out.view(shape)

        return out

```
### 2. Ivy's existing code for torch.bincount

BACKEND FUNCTIONAL API
- [ivy/functional/ivy/experimental/statistical.py](https://github.com/unifyai/ivy/blob/6f2a9ba73e886c7665a27982f0ee7e4ef8db8db9/ivy/functional/ivy/experimental/statistical.py#L268)
- [ivy/functional/backends/tensorflow/experimental/statistical.py](https://github.com/unifyai/ivy/blob/6f2a9ba73e886c7665a27982f0ee7e4ef8db8db9/ivy/functional/backends/tensorflow/experimental/statistical.py#L102)
- [ivy/functional/backends/jax/experimental/statistical.py](https://github.com/unifyai/ivy/blob/6f2a9ba73e886c7665a27982f0ee7e4ef8db8db9/ivy/functional/backends/jax/experimental/statistical.py#L82)
- [ivy/functional/backends/torch/experimental/statistical.py](https://github.com/unifyai/ivy/blob/6f2a9ba73e886c7665a27982f0ee7e4ef8db8db9/ivy/functional/backends/torch/experimental/statistical.py#L118)
- [ivy/functional/backends/numpy/experimental/statistical.py](https://github.com/unifyai/ivy/blob/6f2a9ba73e886c7665a27982f0ee7e4ef8db8db9/ivy/functional/backends/numpy/experimental/statistical.py#L93)

DATA CLASSES
- [ivy/data_classes/container/experimental/statistical.py](https://github.com/unifyai/ivy/blob/6f2a9ba73e886c7665a27982f0ee7e4ef8db8db9/ivy/data_classes/container/experimental/statistical.py#L695)
- [ivy/data_classes/array/experimental/statistical.py](https://github.com/unifyai/ivy/blob/6f2a9ba73e886c7665a27982f0ee7e4ef8db8db9/ivy/data_classes/array/experimental/statistical.py#L288)

BACKEND TEST FUNCTION
- [ivy_tests/test_ivy/test_functional/test_experimental/test_core/test_statistical.py](https://github.com/unifyai/ivy/blob/6f2a9ba73e886c7665a27982f0ee7e4ef8db8db9/ivy_tests/test_ivy/test_functional/test_experimental/test_core/test_statistical.py#L221)

These are the merged codes of bincount in Ivy's repository ! **But there is no functional wrapper for the frontend of torch to handle torch.bincount**
We will have to implement the codes for frontend in two files direstories in this case !  
1. ivy/functional/frontends/torch/utilities.py **to implement the frontend wrapper**
2. ivy/ivy_tests/test_ivy/test_frontends/test_torch/test_utilities.py **to implement the test case got the frontend wrapper in torch**

### 3. Solution - Implementing Torch's Frontend for torch.bincount

> Created a Pull Request for this issue as soon as I spotted it while writing my proposal: https://github.com/unifyai/ivy/pull/13646  

- code snippet for frontend wrapper in ivy/functional/frontends/torch/utilities.py 
```
@with_unsupported_dtypes({"1.11.0 and below": ("int64",)}, "torch")
@to_ivy_arrays_and_back
def bincount(x, weights=None, minlength=0):
    return ivy.bincount(x, weights=weights, minlength=minlength)

```
- code snippet for test function in ivy/ivy_tests/test_ivy/test_frontends/test_torch/test_utilities.py
```
# bincount
@handle_frontend_test(
    fn_tree="torch.bincount",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=1,
        max_value=2,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=1,
            ),
            key="a_s_d",
        ),
    ),
    test_with_out=st.just(False),
)
def test_torch_utilities_bincount(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        weights=None,
        minlength=0
    )

```




