# Sphinx Documentation Builder Guide

This guide gives instruction on hwo to set up and building Sphinx documentation, especially when `.rst` (reStructuredText) files are already created.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Use the Makefile](#how-to-use-the-makefile)
- [Using Sphinx with Existing `.rst` Files](#using-sphinx-with-existing-rst-files)
- [Editing `.rst` Files](#tips-for-editing-rst-files)

## Introduction
Sphinx is a tool for creating intelligent auto-documentation for Python projects. Sphinx uses reStructuredText (`.rst`) files to create documentation. This guide will walk you through the basics of setting up Sphinx, especially when you already have `.rst` files.

## Requirements

- Ensure that [Sphinx](https://www.sphinx-doc.org/) is installed on your system.
- The `sphinx-build` command should either be available in your PATH or the `SPHINXBUILD` environment variable should point to its location.

## Installation

To set up Sphinx and the required extensions for this porject use the following commands: 
    
    ```bash
    pip install sphinx sphinx_rtd_theme myst-parser autodocsumm sphinxcontrib-bibtex sphinx-needs sphinxcontrib-plantuml

    ```

## How to Use the Makefile

1. **Navigate to the Directory**: Open a command prompt or terminal and navigate to the directory containing the make file and the .rst files, in our case: `docs`

2. **Build Documentation**:
   ```bash
   .\make <command> # for Windows: .\make (makefile_name) html 
   ```

3. **View Documentation**: The built html will be in  `_build` directory. Navigate to `_build/html` and open `index.html` in a web browser.

## Using Sphinx with Existing `.rst` Files

1. **Edit the `conf.py` file**: You can edit `conf.py` file - which should be in the same directory as the `.rst` files. This file will contain the configuration for the Sphinx documentation builder.

2. **Working with the `index.rst` file**: This file will contain the table of contents for the documentation. It will also contain the names of the `.rst` files that will be included in the documentation.

3. **Changing `.rst` files**: You can edit the `.rst` files to add content to the documentation. You can also add new `.rst` files to the directory and include them in the `index.rst` file.

## Tips for Editing `.rst` Files
It is important that the `.rst` files are formatted correctly. Here are some tips for editing `.rst` files:

- Using `.. toctree::` directive: This directive is used to create a table of contents for the documentation. It should be used in `index.rst` file to list the `.rst` files that will be included in the documentation. It can also be used in other `.rst` files to create sub-tables of contents. It is also recursively used to create sub-tables of contents within sub-tables of contents.

- Headings:  `=` and `-` symbols create headings. The `=` symbol creates a top-level heading and the `-` symbol creates a second-level heading.

- Links:  `.. _<link_name>: <link>` directive to create a link. To use the link, use the `:ref:` directive. For example, if the link name is `link_name`, then use `:ref:`link_name`` to create a link to the URL.

- Emphasis: Use the `*` symbol to create emphasis. For example, `*emphasis*` will create *emphasis*.

- Images:  `.. image:: <image_path>` to add an image. The `image_path` is the path to the image file. For example, `.. image:: images/image.png` will add the image `image.png` to the documentation.

- Include Files:  `.. include:: <file_path>` directive to include file. For example, `.. include:: file.rst` will include the file `file.rst` in the documentation.