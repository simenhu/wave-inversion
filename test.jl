using PkgTemplates

t = Template(; 
user="simenhu",
dir="",
authors="Acme Corp",
julia=v"1.1",
plugins=[
    License(; name="MPL"),
    Git(; manifest=true, ssh=true),
    GitHubActions(; x86=true),
    Codecov(),
    Documenter{GitHubActions}(),
    Develop(),
],
)