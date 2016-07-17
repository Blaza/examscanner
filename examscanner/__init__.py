"""
The aim of this project is to help teaching assistants at the Faculty of
mathematics in Belgrade do the tedious paperwork regarding entering the results
of the exam and publishing them.

The examscanner package handles the reading of the student id number and the
points scored on the exam from the standard notebook used for exams in the
faculty. It relies on computer vision methods provided by the opencv package.

We have several modules which deal with different parts of the workflow, which
can be depicted like:

Prepare image -> Locate inputs -> Read numbers

1. Preparation is handled by the esutils module functions
2. Location of inputs is done in the locator module
3. Number reading is handled by the reader module

Each part of the process will be described in the appropriate module.
"""
