CMAKE_MINIMUM_REQUIRED(VERSION 4.1)

PROJECT(ABCToFlat)

OPTION(ALEMBIC_ROOT "Install prefix for Alembic installation")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


# FindAlembic etc
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

SET(LIB_TYPE SHARED)
IF (WIN32)
    ADD_DEFINITIONS(-DALEMBIC_DLL)
ENDIF()

# Find Imath or IlmBase.
INCLUDE("cmake/AlembicIlmBase.cmake")

FIND_PACKAGE(Alembic REQUIRED)

ADD_EXECUTABLE(abc-parse abc-parse.cpp)
TARGET_LINK_LIBRARIES(abc-parse ${ALEMBIC_LIB} Imath::ImathConfig)
target_include_directories(abc-parse PRIVATE ${ALEMBIC_INCLUDE_DIR})
set_target_properties(abc-parse PROPERTIES
        BUILD_RPATH ${ALEMBIC_LIBDIR}
        INSTALL_RPATH ${ALEMBIC_LIBDIR})

ADD_EXECUTABLE(abc-separate abc-separate.cpp)
TARGET_LINK_LIBRARIES(abc-separate ${ALEMBIC_LIB} Imath::ImathConfig)
target_include_directories(abc-separate PRIVATE ${ALEMBIC_INCLUDE_DIR})
set_target_properties(abc-separate PROPERTIES
        BUILD_RPATH ${ALEMBIC_LIBDIR}
        INSTALL_RPATH ${ALEMBIC_LIBDIR})

ADD_EXECUTABLE(abc-combine abc-combine.cpp)
TARGET_LINK_LIBRARIES(abc-combine ${ALEMBIC_LIB} Imath::ImathConfig)
target_include_directories(abc-combine PRIVATE ${ALEMBIC_INCLUDE_DIR})
set_target_properties(abc-combine PROPERTIES
        BUILD_RPATH ${ALEMBIC_LIBDIR}
        INSTALL_RPATH ${ALEMBIC_LIBDIR})

ADD_EXECUTABLE(abc-compare abc-compare.cpp)
TARGET_LINK_LIBRARIES(abc-compare ${ALEMBIC_LIB} Imath::ImathConfig)
target_include_directories(abc-compare PRIVATE ${ALEMBIC_INCLUDE_DIR})
set_target_properties(abc-compare PROPERTIES
        BUILD_RPATH ${ALEMBIC_LIBDIR}
        INSTALL_RPATH ${ALEMBIC_LIBDIR})

INSTALL(TARGETS
        abc-parse
        abc-separate
        abc-combine
        abc-compare
)
INSTALL(PROGRAMS abc-separate.py abc-combine.py
        TYPE BIN)
