from setuptools import setup, find_packages

setup(
    name='lstmfm',
    packages=find_packages(),
    version='0.0.1',
    install_requires=[          # 添加了依赖的 package
        'pandas>=0.24',
        'tensorflow==1.14'
    ],
    author="gaoxc",
    author_email="gaoxc0315@outlook.com",
    description="This is an Example Package",
    keywords="hello world example examples",
    url="http://example.com/HelloWorld/",   # project home page, if any
    project_urls={
        "Documentation": "https://docs.example.com/HelloWorld/",
        "Source Code": "https://code.example.com/HelloWorld/",
    },
    classifiers=[
        'License :: OSI Approved :: Python Software Foundation License'
    ]
)