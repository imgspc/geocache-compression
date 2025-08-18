CMAKE_MINIMUM_REQUIRED(VERSION 4.1)

PROJECT(ABCToFlat)

OPTION(ALEMBIC_ROOT "Install prefix for Alembic installation")

# FindAlembic etc
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

SET(LIB_TYPE SHARED)
IF (WIN32)
    ADD_DEFINITIONS(-DALEMBIC_DLL)
ENDIF()

# Find Imath or IlmBase.
INCLUDE("cmake/AlembicIlmBase.cmake")

FIND_PACKAGE(Alembic REQUIRED)

ADD_EXECUTABLE(abc-info alembic-to-flat.cpp)
TARGET_LINK_LIBRARIES(abc-info ${ALEMBIC_LIB} Imath::ImathConfig)
target_include_directories(abc-info PRIVATE ${ALEMBIC_INCLUDE_DIR})

ADD_EXECUTABLE(abc-separate abc-separate.cpp)
TARGET_LINK_LIBRARIES(abc-separate ${ALEMBIC_LIB} Imath::ImathConfig)
target_include_directories(abc-separate PRIVATE ${ALEMBIC_INCLUDE_DIR})
