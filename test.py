from tkinter import *
from tkinter import messagebox
import mysql.connector  # type: ignore

root = Tk()
root.title("FORM KANTORS")

db = mysql.connector.connect(
    host="localhost", user="root", passwd="", database="office"
)

cursor = db.cursor()

main_frame = Frame(root)
main_frame.pack(fill=BOTH, expand=1)

canvas = Canvas(main_frame)
canvas.pack(side=LEFT, fill=BOTH, expand=1)

scrollbar_y = Scrollbar(main_frame, orient=VERTICAL, command=canvas.yview)
scrollbar_y.pack(side=RIGHT, fill=Y)

scrollbar_x = Scrollbar(root, orient=HORIZONTAL, command=canvas.xview)
scrollbar_x.pack(side=BOTTOM, fill=X)

canvas.configure(yscrollcommand=scrollbar_y.set)
canvas.configure(xscrollcommand=scrollbar_x.set)

second_frame = Frame(canvas)

canvas.create_window((0, 0), window=second_frame, anchor="nw")


def load_data():
    for widget in second_frame.winfo_children():
        widget.destroy()

    cursor.execute("SELECT * FROM kantors")
    results = cursor.fetchall()

    no = 1
    for data in results:

        Label(second_frame, text="id", borderwidth=1, relief=SUNKEN, width=5).grid(
            row=0
        )
        Label(second_frame, text="perda", borderwidth=1, relief=SUNKEN, width=15).grid(
            row=0, column=1
        )
        Label(
            second_frame, text="nm_perda", borderwidth=1, relief=SUNKEN, width=15
        ).grid(row=0, column=2)
        Label(second_frame, text="smart", borderwidth=1, relief=SUNKEN, width=15).grid(
            row=0, column=3
        )
        Label(
            second_frame, text="kantor_jenis_id", borderwidth=1, relief=SUNKEN, width=15
        ).grid(row=0, column=4)
        Label(
            second_frame,
            text="singkatan_kantor",
            borderwidth=1,
            relief=SUNKEN,
            width=15,
        ).grid(row=0, column=5)
        Label(
            second_frame, text="nm_kantor", borderwidth=1, relief=SUNKEN, width=15
        ).grid(row=0, column=6)
        Label(
            second_frame, text="nm_smart", borderwidth=1, relief=SUNKEN, width=15
        ).grid(row=0, column=7)
        Label(second_frame, text="alamat", borderwidth=1, relief=SUNKEN, width=15).grid(
            row=0, column=8
        )
        Label(second_frame, text="kota", borderwidth=1, relief=SUNKEN, width=15).grid(
            row=0, column=9
        )
        Label(second_frame, text="telp", borderwidth=1, relief=SUNKEN, width=15).grid(
            row=0, column=10
        )
        Label(second_frame, text="email", borderwidth=1, relief=SUNKEN, width=15).grid(
            row=0, column=11
        )
        Label(
            second_frame, text="website", borderwidth=1, relief=SUNKEN, width=15
        ).grid(row=0, column=12)
        Label(second_frame, text="status", borderwidth=1, relief=SUNKEN, width=15).grid(
            row=0, column=13
        )
        Label(
            second_frame, text="jabatan_id", borderwidth=1, relief=SUNKEN, width=15
        ).grid(row=0, column=14)
        Label(second_frame, text="Aksi", borderwidth=1, relief=SUNKEN, width=15).grid(
            row=0, column=15
        )

        Label(second_frame, text=no, borderwidth=1, relief=SUNKEN, width=5).grid(row=no)
        for col, value in enumerate(data):
            Label(
                second_frame, text=value, borderwidth=1, relief=SUNKEN, width=20
            ).grid(row=no, column=col + 1)
        Button(second_frame, text="DELETE", command=lambda id=data[0]: delete(id)).grid(
            row=no, column=len(data) + 1
        )
        no += 1

    global entry_row
    entry_row = no

    id.grid(column=1, row=entry_row, padx=5, pady=5, in_=second_frame)
    perda.grid(column=2, row=entry_row, padx=5, pady=5, in_=second_frame)
    nm_perda.grid(column=3, row=entry_row, padx=5, pady=5, in_=second_frame)
    smart.grid(column=4, row=entry_row, padx=5, pady=5, in_=second_frame)
    kantor_jenis_id.grid(column=5, row=entry_row, padx=5, pady=5, in_=second_frame)
    singkatan_kantor.grid(column=6, row=entry_row, padx=5, pady=5, in_=second_frame)
    nm_smart.grid(column=7, row=entry_row, padx=5, pady=5, in_=second_frame)
    alamat.grid(column=8, row=entry_row, padx=5, pady=5, in_=second_frame)
    kota.grid(column=9, row=entry_row, padx=5, pady=5, in_=second_frame)
    telp.grid(column=10, row=entry_row, padx=5, pady=5, in_=second_frame)
    email.grid(column=11, row=entry_row, padx=5, pady=5, in_=second_frame)
    website.grid(column=12, row=entry_row, padx=5, pady=5, in_=second_frame)
    status.grid(column=13, row=entry_row, padx=5, pady=5, in_=second_frame)
    jabatan_id.grid(column=14, row=entry_row, padx=5, pady=5, in_=second_frame)
    insert_button.grid(column=1, row=entry_row + 1, padx=5, pady=5, in_=second_frame)
    update_button.grid(column=2, row=entry_row + 1, padx=5, pady=5, in_=second_frame)
    search_button.grid(column=3, row=entry_row + 1, padx=5, pady=5, in_=second_frame)

    second_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))


def insert():
    insert_query = """
    INSERT INTO kantors (id, perda, nm_perda, smart, kantor_jenis_id, singkatan_kantor, nm_kantor, nm_smart, alamat, email, website , status, jabatan_id) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s )
    """
    data = (
        id.get(),
        perda.get(),
        nm_perda.get(),
        smart.get(),
        kantor_jenis_id.get(),
        singkatan_kantor.get(),
        nm_kantor.get(),
        nm_smart.get(),
        alamat.get(),
        email.get(),
        website.get(),
        status.get(),
        jabatan_id.get(),
    )
    cursor.execute, (insert_query, data)
    db.commit()
    messagebox.showinfo("Info", "Data berhasil disimpan")
    load_data()
    clear_form()


def delete(id):
    cursor = db.cursor()
    sql = "DELETE FROM kantors WHERE id=%s"
    val = (id,)
    cursor.execute(
        sql,
        val,
    )
    db.commit()
    messagebox.showinfo("Info", "Data berhasil dihapus")

    load_data()


def update_data(id):
    cursor = db.cursor()
    sql = "UPDATE kantors SET perda=%s, nm_perda=%s, smart=%s, kantor_jenis_id=%s, singkatan_kantor=%s, nm_kantor=%s, nm_smart=%s, alamat=%s, kota=%s, telp=%s, email=%s, website=%s, status=%s, jabatan_id=%s WHERE id=%s" 
    val = (
        perda.get(),
        nm_perda.get(),
        smart.get(),
        kantor_jenis_id.get(),
        singkatan_kantor.get(),
        nm_kantor.get(),
        nm_smart.get(),
        alamat.get(),
        kota.get(),
        telp.get(),
        email.get(),
        website.get(),
        status.get(),
        jabatan_id.get(),
        id
    )
    cursor.execute(sql, val)
    db.commit()
    messagebox.showinfo("Info", "Data berhasil diedit")
    load_data()
    clear_form()


def searchData():
    user_id = id.get()

    # clear fields
    clear_form()

    # search
    search_query = "SELECT * FROM kantors WHERE id =" + user_id
    cursor.execute(search_query)
    userdata = cursor.fetchone()
    id.insert(0, userdata[0])
    perda.insert(0, userdata[1])
    nm_perda.insert(0, userdata[2])
    smart.insert(0, userdata[3])
    kantor_jenis_id.insert(0, userdata[4])
    singkatan_kantor.insert(0, userdata[5])
    nm_kantor.insert(0, userdata[6])
    nm_smart.insert(0, userdata[7])
    alamat.insert(0, userdata[8])
    kota.insert(0, userdata[9])
    telp.insert(0, userdata[10])
    email.insert(0, userdata[11])
    website.insert(0, userdata[12])
    status.insert(0, userdata[13])
    jabatan_id.insert(0, userdata[14])
    messagebox.showinfo("Info", "Data berhasil ditemukan")


def clear_form():
    id.delete(0, "end")
    perda.delete(0, "end")
    nm_perda.delete(0, "end")
    smart.delete(0, "end")
    kantor_jenis_id.delete(0, "end")
    singkatan_kantor.delete(0, "end")
    nm_kantor.delete(0, "end")
    nm_smart.delete(0, "end")
    alamat.delete(0, "end")
    kota.delete(0, "end")
    telp.delete(0, "end")
    email.delete(0, "end")
    website.delete(0, "end")
    status.delete(0, "end")
    jabatan_id.delete(0, "end")


id = Entry(root)
perda = Entry(root)
nm_perda = Entry(root)
smart = Entry(root)
kantor_jenis_id = Entry(root)
singkatan_kantor = Entry(root)
nm_kantor = Entry(root)
nm_smart = Entry(root)
alamat = Entry(root)
kota = Entry(root)
telp = Entry(root)
email = Entry(root)
website = Entry(root)
status = Entry(root)
jabatan_id = Entry(root)

insert_button = Button(root, text="Insert", command=insert)
update_button = Button(root, text="Update", command=update_data)
search_button = Button(root, text="Search", command=searchData)

load_data()
root.mainloop()
