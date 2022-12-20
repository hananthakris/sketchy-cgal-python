# sketchy-cgal-python
Implementation of the sketchy CGAL algorithm in Python
## Running the project
- `python -m venv myenv`
- `source myenv/bin/activate`
- `pip install -r requirements.txt`

- `python main.py --data_store /Users/haritha/Desktop/SketchyCGAL/FilesMaxcut/data/ --R 10 --max_it 6`

## Modifying the configs. 
- Download the `.mat` files from `https://www.cise.ufl.edu/research/sparse/matrices/Gset/index.html`
- Change the command line args R, max_it, and the C matrix if necessary.

## Documentation of the functionalities of sketchy-cgal
- Class variables 
  - 
        z                                                     
        a
        b
        n 
        R
        T
        beta0
        K
        y0
        y
        pob
        stoptol
        FLAG_INCLUSION
        FLAG_LANCZOS
        FLAG_TRACECORRECTION
        FLAG_CAREFULLSTOPPING
        FLAG_METHOD
        FLAG_EVALSURROGATEGAP
        FLAG_MULTIRANK_P1
        FLAG_MULTIRANK_P3
        SKETCH_FIELD
        errfunc
        SCALE_A
        SCALE_C
        SCALE_X
        NORM_A
        SAVEHIST
      
- Functions - 
  - `scale` - Scales the matrices A,X,C according to the appropriate scaling factors
  - `getObjCond` - Get the objective condition
  - `check_stopping_criteria` - Checks when the algorithm should stop, given a time step.
  - `create_err_structs` - Creates the final outputs.
  - `updateErrStructs` - Updates the output for each timestep in SAVEHIST.
  - `solve` - The Sketchy-CGAL solver.
