del .\rl\results\logs_\* /S /Q
FOR /D %%p IN (".\rl\results\logs_\*") DO rmdir "%%p" /s /q