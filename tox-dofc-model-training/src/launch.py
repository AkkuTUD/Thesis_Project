import sys

sys.path.append('yolov5')
sys.argv = sys.argv[1:]
main=open(sys.argv[0]).read()
print("launch.py: adjust tqdm progress bar")

from tqdm import tqdm
from functools import partialmethod

# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

#t_exit = tqdm.__exit__
#t_refresh = tqdm.refresh

#def __exit__(self, *exc):
#    global t_exit
#    global t_refresh
#    self.refresh = t_refresh
#    self.t_refresh()
#    t_exit(self, *exc)

#def refresh(self, nolock=False, lock_args=None):
#    pass

#tqdm.__exit__ = __exit__
#tqdm.refresh = refresh

exec(main)
