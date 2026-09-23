# ECI and fitness with supply chain information and import dependencies

## Project structure

```
├── src/
│   ├── algorithms.py
│   ├── preprocessing.py
│   ├── ...
│   ├── ...
│   └── ...
├── notebooks/
│   └── ...
├── README.md
└── requirements.txt
```

## `src/` folder

Contains the modules with the functions used in the project. Specifically:

- **`algorithms.py`** — core module, contains the various ECI and fitness (with relative extension) algorithms.
- **`preprocessing.py`** — core module, contains the preprocessing pipelines to work effectively on data.
- The other modules in this folder contain supporting or experimental functions, not systematically organized and not consistently used across the project. They should be considered accessory, not part of the stable "core" codebase.

 

## `notebooks/` folder

Contains Jupyter notebooks used mainly for quick experiments, tests, and ad-hoc exploration. Some notes:

- They are not systematically organized.
- They get created, modified, and often **deleted** once no longer needed.
- **They should not be considered a stable or reliable resource of the project** — their content can change or disappear without notice.

For reusable, stable code, always refer to the `src/` folder.

## How to use the project



## Project status

Actively under development. The organization of the accessory modules in `src/` may be revised in the future.

## Author
Vittorio Bargero



