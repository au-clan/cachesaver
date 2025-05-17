import re
import random
from typing import Tuple, Optional, List
import subprocess
import json
import signal
import contextlib
import io
import tempfile
import os
import faulthandler
import platform
import multiprocessing


from .state import StateHumanEval
from ...typedefs import Environment, MAX_SEED

TIMEOUT = 5.0

class EnvironmentHumanEval(Environment):

    @staticmethod
    def step(state: StateHumanEval, action: str) -> StateHumanEval:
        """
        Takes a step in the environment based on the given action.
        """

        # Parse the action
        completion = parse_action(action)

        # randomness handling
        random.seed(state.randomness)
        randomness = random.randint(0, MAX_SEED)

        state = StateHumanEval(
            puzzle=state.puzzle,
            current_state=completion,
            steps=state.steps + [completion],
            entry_point=state.entry_point,
            test=state.test,
            randomness=randomness
        )
        return state

    @staticmethod
    def is_valid(state: StateHumanEval, action: str) -> bool:
        """
        Checks if the action taken is valid.
        """
        raise NotImplementedError("Action validation logic is not implemented.")
    
    @staticmethod
    def is_final(state: StateHumanEval) -> bool:
        """
        Checks if the current state is a final state.
        """
        return True
    
    @staticmethod
    def evaluate(state: StateHumanEval) -> Tuple[bool, float]:
        """
        Evaluates the current state.
        """
        # Extract the code and test from the state
        code = state.current_state
        entry_point = state.entry_point
        test = state.test

        # Evaluate the code
        if "def" in code:
            # Python code
            solved, score = evaluate_code_python(code, entry_point, test)
        else:
            # Rust code
            solved, score = evaluate_code_rust(code, entry_point, test)
        return solved, score
        


#---Helper functions---#
def parse_action(string) -> str | None:
    pattern = r'```[^`]+```'
    match = re.match(pattern, string)
    return "\n".join(match.group(0).split('\n')[1:-1]) if match else string

def evaluate_code_python(code: str, entry_point: str, test: str) -> Tuple[bool, float]: # NOTE: Only works on a UNIX system as we are using signal. Need to change this to use a different method for Windows or general case if needed.
    """
    Evaluates the given code using the provided entry point and test.
    """
    manager = multiprocessing.Manager()
    result = manager.list()
    ts = separate_tests(test)
    for t in ts:
        p = multiprocessing.Process(target=unsafe_execute, args=(code, entry_point, t, TIMEOUT, result))
        p.start()
        p.join(timeout=TIMEOUT+1)
        if p.is_alive():
            p.kill()

    if not result:
        return False, 0.0
    
    else:
        return True, sum(result) / len(result)
    
def evaluate_code_rust(code: str, entry_point: str, test: str) -> Tuple[bool, float]:
    """
    Evaluates the given code using the provided entry point and test.
    """
    # This implementation is from Reflexion, it uses rust compiler to compile the code and then run it.

    def cleanup():
        # Clean up the temporary directory
        if os.path.exists(tmp_dir):
            os.system(f"rm -rf {tmp_dir}")


    tmp_dir, tmp_path = create_temp_project()
    write_to_file_toplevel(tmp_path, code + test)

    res = run_with_timeout(
        "cargo check --message-format=json", tmp_dir, timeout=TIMEOUT)
    
    if res is None:
        cleanup()
        return False, 0.0

    errs = grab_compile_errs(res[0])
    if len(errs) > 0:
        cleanup()
        return True, 0.0

    res = run_with_timeout("cargo run", tmp_dir, timeout=TIMEOUT)
    cleanup()

    if res is None:
        return False, 0.0
    else:
        errs = grab_runtime_errs(res[0] + "\n" + res[1])
        if len(errs) > 0:
            return True, 0.0
        
        return True, 1.0
    

def separate_tests(test):
    parts = test.split("def check(candidate):")
    tests = parts[-1].strip().split("\n")
    tests = [parts[0] + f"""def check(candidate):
    {t.strip()}""".strip() for t in tests]
    return tests

#---Context Managers---#
# From HumanEval repo for evaluating code with timelimit and a bit more securely. Still recommending to do it inside a sandbox.
def unsafe_execute(code: str, entry_point: str, test: str, timeout: float, result):
    with create_tempdir():
        import os
        import shutil
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        reliability_guard()

        program = (
            code 
            + '\n'
            + test
            + '\n'
            + f'check({entry_point})'
        )

        try:
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(program, exec_globals)
            result.append(1)
        except Exception:
            result.append(0)

        # Clean up
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield

@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname

class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False
    
class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"

@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)

def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


#---Helper function for Rust code evaluation---#
def create_temp_project() -> Tuple[str, str]:
    # get pid of the process
    pid = os.getpid()
    # get random number
    rand = os.urandom(8).hex()
    # create a temp directory
    temp_dir = f"/tmp/cargo_harness-{pid}-{rand}"
    # delete the temp directory if it exists
    if os.path.exists(temp_dir):
        os.system(f"rm -rf {temp_dir}")
    os.mkdir(temp_dir)
    # move the cargo harness into the temp directory
    # os.system(f"cp -r {cargo_harness_dir}/* {temp_dir}")

    # Create a Cargo.toml file if it's missing
    cargo_toml_content = """\
    [package]
    name = "cargo_harness"
    version = "0.1.0"
    edition = "2021"

    # See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

    [dependencies]
    """
    cargo_toml_path = os.path.join(temp_dir, "Cargo.toml")
    with open(cargo_toml_path, "w") as f:
        f.write(cargo_toml_content)

    main_path = os.path.join(temp_dir, "src", "main.rs")
    return temp_dir, main_path

class CompileErr:
    def __init__(self, rendered):
        self.rendered = rendered

    def __str__(self):
        return self.rendered

    def __repr__(self):
        return "{" + str(self) + "}"
    
class RuntimeErr:
    def __init__(self, left, right, line, column, panic_reason):
        # right and left are only used for assert_eq! errors
        self.left = left
        self.right = right
        # NOTE: currently not using the below
        self.line = line
        self.column = column
        self.panic_reason = panic_reason

    def __str__(self):
        if self.left is not None and self.right is not None:
            return f"assertion failed: {self.left} == {self.right}"
        else:
            return self.panic_reason

    def __repr__(self):
        return "{" + str(self) + "}"
    
def write_to_file_toplevel(path: str, code: str):
    # delete the file if it exists
    if os.path.exists(path):
        os.remove(path)
    # write the code to the file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(code)

def run_with_timeout(cmd: str, tmp_cargo_path: str, timeout: float = 5.0, print_debug: bool = False) -> Optional[Tuple[str, str]]:
    """
    Runs the given command with a timeout. Produces a tuple of stdout and stderr.
    If the command times out, returns None.
    """
    import os
    import shutil
    rmtree = shutil.rmtree
    rmdir = os.rmdir
    chdir = os.chdir

    try:
        with swallow_io():
            with time_limit(timeout):
                # run the command
                p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, cwd=tmp_cargo_path)
                out, err = p.communicate()
    except Exception:
        p.kill()
        return None
    
    # if p.returncode != 0:
    #     return None 

    # decode the output
    out = out.decode("utf-8")
    err = err.decode("utf-8")
    if print_debug:
        print("## RUN OUTPUTS ##")
        print("STDOUT:")
        print(out)
        print("STDERR:")
        print(err, flush=True)
    
    # Clean up
    shutil.rmtree = rmtree
    os.rmdir = rmdir
    os.chdir = chdir

    return out, err

# assumes that the input is the stdout of cargo check --message-format=json
# returns a list of compile errors as CompileErr objects
def grab_compile_errs(inp: str) -> List[CompileErr]:
    # we get a stream of json objects, so we need to parse them one by one
    objs = []
    for line in inp.splitlines():
        if line == "":
            continue
        o = json.loads(line)
        if o is not None and o["reason"] == "compiler-message" and \
                o["message"]["level"] == "error" and \
                o["message"]["spans"] != []:
            rendered = o["message"]["rendered"]
            objs.append(CompileErr(rendered))

    return objs

def grab_runtime_errs(inp: str) -> List[RuntimeErr]:
    failed_asserts = []
    curr_left = None
    panic_reason = None

    for line in inp.splitlines():
        if "fatal runtime" in line:
            # we have a panic
            panic_idx = line.index("fatal runtime")
            panic_reason = line[panic_idx + len("fatal runtime") + 1:]
        elif "panicked at" in line:
            panic_idx = line.index("panicked at")
            # strip source line if it exists
            if "src/main.rs" in line:
                line = line[:line.index("src/main.rs")]
            panic_reason = line[panic_idx + len("panicked at") + 1:]
        elif "panic" in line:
            panic_idx = line.index("panic")
            panic_reason = line[panic_idx:]
        elif "left:" in line:
            split = line.split("`")
            if len(split) < 2:
                continue
            curr_left = split[1]
        elif "right:" in line:
            split = line.split("`")
            if len(split) < 2:
                continue
            curr_right = split[1]
            # get the line and column number
            fileinto = line.split(",")[-1]
            line = int(fileinto.split(":")[1])
            column = int(fileinto.split(":")[2])
            failed_asserts.append(RuntimeErr(
                curr_left, curr_right, line, column, panic_reason))
            curr_left = None
            panic_reason = None

    if panic_reason is not None:
        failed_asserts.append(RuntimeErr(None, None, None, None, panic_reason))

    return failed_asserts