# Proof of Concept for Kornia - Transpiler Project for Ivy

1. [My understanding with Ivy's transpiler](https://github.com/harshxtanwar/harshwardhantanwarPOCkornia#my-understanding-with-ivys-transpiler)
   - [Transpiling functions eagerly](https://github.com/harshxtanwar/harshwardhantanwarPOCkornia/blob/main/README.md#1-transpiling-function-eagerly)
   - [Ivy transpiling Functions, Models and Frameworks !](https://github.com/harshxtanwar/harshwardhantanwarPOCkornia/blob/main/README.md#2-ivy-transpiling-functions-models-and-frameworks-)
2. [Objective of the project](https://github.com/harshxtanwar/harshwardhantanwarPOCkornia/blob/main/README.md#objective-of-the-project)
   - [Main Objective](https://github.com/harshxtanwar/harshwardhantanwarPOCkornia/blob/main/README.md#1-main-objective)
   - [Workflow and how exactly I will proceed with the project](https://github.com/harshxtanwar/harshwardhantanwarPOCkornia#2-workflow-and-how-exactly-i-will-proceed-with-the-project)
3. [Implementation Example, Missing Torch Frontend](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#implementation-example-missing-torch-frontend)
   - [Pytorch Geometric's Code](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#1-pytorch-geometrics-code)
   - [Ivy's existing code for torch.bincount](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#2-ivys-existing-code-for-torchbincount)
   - [Solution](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#3-solution---implementing-torchs-frontend-for-torchbincount)

## My understanding with Ivy's transpiler
In this section, I aim to convey to you my understanding of Ivy's transpiler. I will talk about how
Ivy transpiles functions, modules and frameworks eagerly and lazily !

### 1. Transpiling function eagerly
- step 1: A graph is compiled for the function that Ivy is trying to transpile in the framework that the function is in.  
          For example: in case of a kornia function, we will get a Torch graph.  
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
- The main aim of this project is to make **Kornia** compatible with all other machine learning frameworks supported by Ivy  
like Numpy, Jax, Paddle and Tensorflow.
- After the successful implementation of this project, users of kornia will be able to transpile the whole 
kornia module to a framework of their choice !
- The users will highly benefit from this as the runtime efficiency is greatly improved when using a JAX backend on TPU, compared  
to the original PyTorch implementation.

### 2. Workflow and how exactly I will proceed with the project
Below are the steps I will follow to work on the project in a chronological order.
- Finding and creating a list of all of the pyTorch function used in the kornia's Module.
- Eliminating the functions from the list which already have been implemented in Ivy's both backend and frontend
- Creating a list of functions that already have been implemented in the backend but don't have any frontend wrapper around them
and creating another list of functions that need to be implemented for all backends in Ivy along with a frontend wrapper.
- After finalising the list, I will start working with multiple pull requests to first finish writing codes for the functions with **missing frontend wrappers** in Ivy's repository.
- Then I will work on multiple pull requests to implement functions that have **both missing backend handlers and frontend wrapper along with the test cases** in Ivy's repository.
- Once this is done, I will make a pull request to kornia's repository where I will implement kornia's framework handlers to enable transpilation to provide kornia's users with functions that can transpile kornia to a framework of their choice.
- I will create Google Colab Demos showcasing how a set of differentiable computer vision operations can be done using kornia can be used in TensorFlow and JAX projects (or any other framework) for the users of kornia.
- I will also create Google Colab Demos showing how the runtime efficiency is greatly improved when using a JAX backend on TPU, compared to the original PyTorch implementation.

## Implementation Example, Missing Torch Frontend

torch.bincount function is one such function which is used in Kornia, in the 
kornia/kornia/metrics/confusion_matrix.py directory and you can view the use of torch.bincount
function in this directory at this [link]([https://github.com/pyg-team/pytorch_geometric/blob/7469edee6edae1afd8a9dc61b1494ec6412195aa/torch_geometric/nn/aggr/quantile.py#L78](https://github.com/kornia/kornia/blob/da2f1325014a158be36d62bdd20293ef97e574f4/kornia/metrics/confusion_matrix.py))


### 1. Pytorch Geometric's Code
View the link above to view the exact location of PyG's functio and torch.bincount in their repository in github

```
import torch

# Inspired by:
# https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py#L68-L73


def confusion_matrix(
    input: torch.Tensor, target: torch.Tensor, num_classes: int, normalized: bool = False
) -> torch.Tensor:
    r"""Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        input: tensor with estimated targets returned by a
          classifier. The shape can be :math:`(B, *)` and must contain integer
          values between 0 and K-1.
        target: tensor with ground truth (correct) target
          values. The shape can be :math:`(B, *)` and must contain integer
          values between 0 and K-1, where targets are assumed to be provided as
          one-hot vectors.
        num_classes: total possible number of classes in target.
        normalized: whether to return the confusion matrix normalized.
    Returns:
        a tensor containing the confusion matrix with shape
        :math:`(B, K, K)` where K is the number of classes.
    Example:
        >>> logits = torch.tensor([[0, 1, 0]])
        >>> target = torch.tensor([[0, 1, 0]])
        >>> confusion_matrix(logits, target, num_classes=3)
        tensor([[[2., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 0.]]])
    """
    if not torch.is_tensor(input) and input.dtype is not torch.int64:
        raise TypeError("Input input type is not a torch.Tensor with " "torch.int64 dtype. Got {}".format(type(input)))

    if not torch.is_tensor(target) and target.dtype is not torch.int64:
        raise TypeError(
            "Input target type is not a torch.Tensor with " "torch.int64 dtype. Got {}".format(type(target))
        )
    if not input.shape == target.shape:
        raise ValueError(
            "Inputs input and target must have the same shape. " "Got: {} and {}".format(input.shape, target.shape)
        )
    if not input.device == target.device:
        raise ValueError("Inputs must be in the same device. " "Got: {} - {}".format(input.device, target.device))

    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("The number of classes must be an integer bigger " "than two. Got: {}".format(num_classes))

    batch_size: int = input.shape[0]

    # hack for bitcounting 2 arrays together
    # NOTE: torch.bincount does not implement batched version
    pre_bincount: torch.Tensor = input + target * num_classes
    pre_bincount_vec: torch.Tensor = pre_bincount.view(batch_size, -1)

    confusion_list = []
    for iter_id in range(batch_size):
        pb: torch.Tensor = pre_bincount_vec[iter_id]
        bin_count: torch.Tensor = torch.bincount(pb, minlength=num_classes**2)
        confusion_list.append(bin_count)

    confusion_vec: torch.Tensor = torch.stack(confusion_list)
    confusion_mat: torch.Tensor = confusion_vec.view(batch_size, num_classes, num_classes).to(torch.float32)  # BxKxK

    if normalized:
        norm_val: torch.Tensor = torch.sum(confusion_mat, dim=1, keepdim=True)
        confusion_mat = confusion_mat / (norm_val + 1e-6)

    return 
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




