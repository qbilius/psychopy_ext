Why use `psychopy_ext`?
=======================

Successful accumulation of knowledge is critically dependent on the ability to verify and
replicate every part of a scientific conduct. Python and its scientific packages have greatly
fostered the ability to share and build upon experimental and analysis code. However, while
open access to publications is immediately perceived as desired, open sourcing experiment and
analysis code is often ignored or met with a grain of skepticism in the neuroscience community,
and for a good reason: many publications would be difficult to reproduce from start to end
given typically poor coding skills, lack of version control habits, and the prevalence of manual
implementation of many tasks (such as statistical analyses or plotting), leading to a reproducible
research in theory but not in practice.

I argue that the primary reason of such unreproducible research is the lack of tools that would
seamlessly enact good coding and sharing standards. Here I propose a framework tailored to
the needs of the neuroscience community that ties together project organization, creation of
experiments, behavioral and functional magnetic resonance imaging (fMRI) data analyses,
and publication quality (i.e., pretty) plotting using a unified and relatively rigid interface.
Unlike *PsychoPy*, *PyMVPA* or *matplotlib* that are very flexible and support multiple options to
suit everyone’s needs, the underlying philosophy of *psychopy_ext*
is to act as the glue at a higher level of operation by choosing reasonable defaults
for these packages and providing patterns for common tasks with a minimal user intervention.

For example, each experiment is expected to be a module with classes in it representing
different parts of scientific conduct (e.g., stimulus presentation or data analysis), and methods
representing an atomic task at hand (e.g., showing experimental instructions or running a
support vector machine analysis). Such organization is not only natural and easy to follow in
an object-oriented environment but also allows an automatic generation of a command line
and graphic user interfaces for customizing and executing these tasks conveniently. Due to a
rigid structure, *psychopy_ext* can more successfully than typical packages address realistic user
cases. For instance, running a support vector machine on fMRI data involves multiple steps of
preprocessing, aggregating over relevant axes, combining results over participants, and, ideally,
unit testing. Since it is impossible to guess the particular configuration at hand, typically the user
has to implement these steps manually. However, thanks to a common design pattern in analyses
deriving from *psychopy_ext*, these operations can be performed seamlessly out of the box.

While these choices might be limiting in certain cases, the aim of *psychopy_ext* is to provide an
intuitive basic framework for building transparent and shareable research projects.

