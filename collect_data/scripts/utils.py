import os

def make_dir(path):
  real_path = os.path.expanduser(path)
  if not os.path.exists(real_path):
    os.makedirs(real_path)
  return real_path
