.. _faq-label:

Frequently asked questions
==========================

.. contents:: Content:
   :depth: 2

Labs
----

1. Argument 1 to "get_top_n" has incompatible type "Dict[str, int]"; expected "Dict[str, Union[int, float]]" [arg-type]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This problem frequently occurs in ``lab_1_keywords_tfidf`` and is easily fixed.

Typically, this remark is followed by 2 notes:

* ``note: "Dict" is invariant -- see [link]``
* ``note: Consider using "Mapping" instead, which is covariant in the value type``

Although error message may not seem to be particularly clear, it is
rather simple to fix it. To solve the problem it is enough to carefully
follow the task description. In task description, students are required
to demonstrate ``get_top_n`` using two dictionaries: the one with
``TF-IDF`` scores and the one with chi-values. Both of those
dictionaries contain ``float`` data as values, and such usage does not
cause any problems.

Issues begin when one decides to use ``get_top_n`` on frequency
dictionary, which is **not** required in task description. Frequency
dictionaries have integer values which does not match very well with the
``get_top_n`` typing in this particular lab. This is why it leads to
``MyPy`` complaining.

To fix this problem, only use ``get_top_n`` on dictionaries with
``float`` values.

2. Cannot find implementation or library stub for module named "main" [import]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let’s say the structure of your project looks like this:

.. code:: text

   +-- 2023-2-level-labs
       +-- config
       +-- docs
       +-- lab_1_keywords_tfidf
           +-- assets
           +-- tests
           +-- main.py
           +-- start.py
           +-- target_score.txt
           +-- README.md
       +-- seminars
   ...

You want to import functions from ``main.py``. To do that, remember that
the checking program looks at your code from the root folder, meaning
that for it the correct name of the ``main.py`` would be the following:
``lab_1_keywords_tfidf/main.py``

This is why to import functions from ``main.py`` in your ``start.py``
you need to put it the following way:

.. code:: text

   from lab_1_keywords_tfidf.main import <functions you want to import>

3. Argument 1 to <function name> Has incompatible type "Optional[<certain type>]"; expected "[<certain type>]"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some of the laboratory works there is a requirement to check input
data. In other words, apart from main logic of the function, one should
verify that all input arguments are of the expected type, and, for
example, return ``None`` otherwise. This is precisely why this ``MyPy``
warning is raised: if in a sequence of two functions the former one can
return ``None`` as an indicator of corrupt data, and the latter one does
not expect ``None`` among correct input values, there is a risk of
passing data that is obviously incorrect.

To avoid this ``MyPy`` remark, it is necessary to check whether the
returned value is not ``None`` before proceeding to feed it to the
second function.

For example, let’s say we have the following two functions. The first
one unites two lists, and the second one sums all the elements in the
list.

.. code:: py

   def function1(arg1: list[int], arg2: list[int]) -> Optional[list[int]]:
       if not arg1 or not arg2:
           return None
       return arg1 + arg2

   def function2(arg: list[int]) -> Optional[int]:
       if not arg:
           return None
       return sum(arg)

We want to use these functions sequentially: firstly we want to unite
two lists, and then find its sum. This is an incorrect way to do that:

.. code:: py

   united_list = function1(list1, list2)
   elements_sum = function2(united_list)

``function1`` can return ``None``, and we must not pass it to
``function2``. Correct way to check it:

.. code:: py

   united_list = function1(list1, list2)
   if united_list:
       elements_sum = function2(united_list)

4. Incompatible types in assignment (expression has type X, variable has type Y)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python is a dynamically typed programming language, meaning that during
execution of a program in Python same variables can be assigned values
of different types. Although it is not prohibited in the language, it
may still be not the best practice. Reusing variables in such a way can
make your code more vulnerable as there would be a higher probability of
making a mistake that is hard to track. This is why ``MyPy`` highlights
such variables: maintaining consistency of typing throughout value
re-assigning should solve this problem.
More about `incompatible re-definitions
<https://mypy.readthedocs.io/en/stable/common_issues.html#redefinitions-with-incompatible-types>`__.
More about `perks of mypy-style static typing
<https://mypy.readthedocs.io/en/stable/faq.html#why-have-both-dynamic-and-static-typing>`__.

5. During working in PyCharm, interpreter cannot be found
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In many cases the issue turns out to be wrong opening of the PyCharm.
Make sure that you open the whole ``202X-2-level-labs`` as a project,
not just the folder with a particular lab.

More details on correct PyCharm opening can be found in :ref:`starting-guide-label`.

Running tests
-------------

1. Why is my CI job cancelled?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usually that happens because your CI check runs for too
long. Possible reasons is that you do not control number of articles
that you collect from your seed URL. If you feel that the problem is
with infrastructure, call a mentor in the group chat.

2. Why is my CI job not started?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usually that happens because your fork has conflicts with a
base repository. Resolve them by merging the upstream, or if it all
sounds new for you, call a mentor in the group chat.
