--OPENCL="/opt/AMDAPP/"
OPENCL="/usr/local/cuda"
OZLIB="OZlib"
NIFTI="/usr/include/nifti"

solution "SignedDistFunc"
 --configurations { "Debug" , "Release"}
 configurations { "Release" , "Debug"}
   --location("build")

   -- A project defines one build target
   project "SignedDistFunc"
   targetdir "dist"

        --Shared are .o static are .a
      kind "ConsoleApp"

      includedirs{--My libraries
                  OZLIB,
                  NIFTI,
                  -- cl.h
                 OPENCL.."/include",
                 "src", "src/headers"
             }

      -- os.copyfile("src/resources/SDF.cl","dist")
      -- os.copyfile("src/resources/SDFVoroBuf.cl","dist")

      libdirs{OZLIB}

      location "."
      language "C++"

      -- Current project files
      files {"src/**.h", "src/**.cpp" }     

      links({"OpenCL","GL","GLU","glut","GLEW","X11","m","FileManager","niftiio",
          "GLManager","CLManager","ImageManager","GordonTimers","freeimage"})
 
      configuration "Debug"
         --defines { "DEBUG" , "PRINT" }
         
         -- For debugging
         --defines { "DEBUG" }
         
         -- For saving the images
         defines { "SAVE" }
         flags { "Symbols" }


      configuration "Release"
         defines { "NDEBUG" }
         flags { "Optimize" }
