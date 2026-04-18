set(MODULE_NAME AnatomyAtlas)

slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_NAME}.py
  RESOURCES
    Resources/UI/${MODULE_NAME}.ui
    Resources/anatomy_data.json
  WITH_GENERIC_TESTS
)

if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT Testing/Python/${MODULE_NAME}Test.py)
endif()
