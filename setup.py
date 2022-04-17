import sys, setuptools

if float("{}.{}".format(sys.version_info[0], sys.version_info[1])) < 3.8:
    sys.stdout.write("chopin requires Python 3.8 or higher. Please update your Python installation")

setuptools.setup(name='chopin',
                 version='1.0',
                 author='Fabio Cumbo',
                 author_email='fabio.cumbo@gmail.com',
                 url='http://github.com/fabio-cumbo/chopin',
                 license='LICENSE',
                 packages=setuptools.find_packages(),
                 entry_points={
                     'console_scripts': ['chopin = src.chopin:chopin']
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