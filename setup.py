import sys, setuptools

if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and  sys.version_info[1] < 8):
    sys.stdout.write("chopin2 requires Python 3.8 or higher. Your Python your current Python version is {}.{}.{}"
                     .format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))

setuptools.setup(name='chopin2',
                 version='1.0.6',
                 author='Fabio Cumbo',
                 author_email='fabio.cumbo@gmail.com',
                 url='http://github.com/fabio-cumbo/chopin2',
                 license='LICENSE',
                 packages=setuptools.find_packages(),
                 entry_points={
                     'console_scripts': ['chopin2 = chopin2.chopin2:chopin2']
                 },
                 description='Supervised Classification with Hyperdimensional Computing',
                 long_description=open('README.md').read(),
                 long_description_content_type='text/markdown',
                 install_requires=[
                     "numpy",
                     "pyspark",
                     "numba"
                 ],
                 zip_safe=False)