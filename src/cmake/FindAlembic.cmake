# Copyright Imaginary Spaces, 2025

SET(_alembic_SEARCH_DIRS
    ${ALEMBIC_ROOT}
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local
    /usr
    /opt/local
    /opt
)

# Find All.h and hope the rest of alembic is there.
find_path(ALEMBIC_INCLUDE_DIR
    NAMES
        Alembic/Abc/All.h
    HINTS
        ${_alembic_SEARCH_DIRS}
    PATH_SUFFIXES
        include
)

find_library(ALEMBIC_LIB
    NAMES
        Alembic
    HINTS
        ${_alembic_SEARCH_DIRS}
    PATH_SUFFIXES
        lib
)

# Default to found, set not-found if part of it wasn't found.
set(ALEMBIC_FOUND TRUE)

if (ALEMBIC_INCLUDE_DIR STREQUAL "ALEMBIC_INCLUDE_DIR-NOTFOUND")
    message("Alembic include path not found, set ALEMBIC_ROOT")
    set(ALEMBIC_FOUND FALSE)
endif()

if (ALEMBIC_LIB STREQUAL "ALEMBIC_LIB-NOTFOUND")
    message("Alembic compiled libraries not found, set ALEMBIC_ROOT")
    set(ALEMBIC_FOUND FALSE)
endif()
