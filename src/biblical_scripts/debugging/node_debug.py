import pdb
import sys
import traceback

from kedro.framework.hooks import hook_impl

class PDBNodeDebugHook:
    """A hook class for creating a post mortem debugging with the PDB debugger
    whenever an error is triggered within a node. The local scope from when the
    exception occured is available within this debugging session.
    """

    @hook_impl
    def on_node_error(self):
        _, _, traceback_object = sys.exc_info()

        #  Print the traceback information for debugging ease
        traceback.print_tb(traceback_object)

        # Drop you into a post mortem debugging session
        pdb.post_mortem(traceback_object)