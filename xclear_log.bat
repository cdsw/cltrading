del .\rlold\results\logs_\* /S /Q
FOR /D %%p IN (".\rlold\results\logs_\*") DO rmdir "%%p" /s /q